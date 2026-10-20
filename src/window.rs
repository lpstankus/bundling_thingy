use crate::{Options, gpu::Gpu, graph::Graph};
use color_eyre::eyre::{OptionExt, Result};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const INPUTS: [&str; 5] = [
    "data/fr-airlines.trl",
    "data/us-migrations.trl",
    "data/us-airlines.trl",
    "data/rj-notlocal-elective.trl",
    "data/rj-2019.trl",
];
struct App {
    opts: Options,
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    graphs: Vec<Graph>,
    index: usize,
    dirty: bool,
    error: Option<color_eyre::eyre::Report>,
}
impl App {
    fn draw(&mut self) -> Result<()> {
        let gpu = self.gpu.as_mut().ok_or_eyre("GPU is not initialized")?;
        let window = self
            .window
            .as_ref()
            .ok_or_eyre("window is not initialized")?;
        if self.dirty {
            gpu.upload(
                &self
                    .graphs
                    .get(self.index)
                    .ok_or_eyre("invalid graph selection")?
                    .mesh(0.002)?,
            )?;
            self.dirty = false;
        }
        gpu.present()?;
        window.set_title(&format!(
            "Bundling Thingy - {}",
            INPUTS.get(self.index).ok_or_eyre("invalid graph title")?
        ));
        Ok(())
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let result = (|| -> Result<()> {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("Bundling Thingy")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1600, 1600)),
                )?,
            );
            self.gpu = Some(pollster::block_on(Gpu::new(
                Some(window.clone()),
                self.opts.validate_gpu,
            ))?);
            for input in INPUTS {
                let mut g = Graph::load(input.as_ref())?;
                g.normalize()?;
                g.resample(0.02)?;
                self.graphs.push(g);
            }
            window.request_redraw();
            self.window = Some(window);
            Ok(())
        })();
        if let Err(e) = result {
            self.error = Some(e);
            event_loop.exit();
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Err(e) = self.handle_event(event_loop, event) {
            self.error = Some(e);
            event_loop.exit();
        }
    }
}
impl App {
    fn handle_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) -> Result<()> {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowRight | NamedKey::Space) => {
                        self.index = self
                            .index
                            .checked_add(1)
                            .and_then(|i| i.checked_rem(self.graphs.len()))
                            .ok_or_eyre("invalid graph selection")?;
                        self.dirty = true;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.index = if self.index == 0 {
                            self.graphs.len().checked_sub(1).ok_or_eyre("no graphs")?
                        } else {
                            self.index.saturating_sub(1)
                        };
                        self.dirty = true;
                    }
                    _ => {}
                }
                if self.dirty {
                    self.window
                        .as_ref()
                        .ok_or_eyre("window is not initialized")?
                        .request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                self.gpu
                    .as_mut()
                    .ok_or_eyre("GPU is not initialized")?
                    .resize(size.width, size.height);
                if size.width > 0 && size.height > 0 {
                    self.window
                        .as_ref()
                        .ok_or_eyre("window is not initialized")?
                        .request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let size = self
                    .window
                    .as_ref()
                    .ok_or_eyre("window is not initialized")?
                    .inner_size();
                if size.width == 0 || size.height == 0 {
                    return Ok(());
                }
                self.draw()?;
            }
            _ => {}
        }
        Ok(())
    }
}
pub fn run(opts: Options) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        opts,
        window: None,
        gpu: None,
        graphs: vec![],
        index: 0,
        dirty: true,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(e) = app.error {
        return Err(e);
    }
    Ok(())
}
