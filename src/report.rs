use color_eyre::eyre::{Result, WrapErr};
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

struct TemporaryReport {
    path: PathBuf,
}

impl Drop for TemporaryReport {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn invalidate(directory: &Path, file: &str) -> Result<()> {
    for path in [directory.join(file), directory.join(format!("{file}.tmp"))] {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).wrap_err_with(|| format!("remove stale {}", path.display()));
            }
        }
    }
    Ok(())
}

pub fn publish<T: serde::Serialize>(directory: &Path, file: &str, report: &T) -> Result<()> {
    let path = directory.join(file);
    let temporary = TemporaryReport {
        path: directory.join(format!("{file}.tmp")),
    };
    let mut writer = std::io::BufWriter::new(
        fs::File::create(&temporary.path)
            .wrap_err_with(|| format!("create {}", temporary.path.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, report).wrap_err("serialize report")?;
    std::io::Write::flush(&mut writer)
        .wrap_err_with(|| format!("flush {}", temporary.path.display()))?;
    drop(writer);
    fs::rename(&temporary.path, &path).wrap_err_with(|| format!("publish {}", path.display()))
}
