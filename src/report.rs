use color_eyre::eyre::{Result, WrapErr};
use std::{fs, io::ErrorKind, path::Path};

pub fn invalidate(directory: &Path, file: &str) -> Result<()> {
    let path = directory.join(file);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).wrap_err_with(|| format!("remove stale {}", path.display())),
    }
}

/// Serialize and flush a temporary file, then rename it into place. A failed
/// publication removes its temporary file and returns the original error.
pub fn publish<T: serde::Serialize>(directory: &Path, file: &str, report: &T) -> Result<()> {
    let path = directory.join(file);
    let temporary = directory.join(format!("{file}.tmp"));
    let result = write(&temporary, report).and_then(|()| {
        fs::rename(&temporary, &path).wrap_err_with(|| format!("publish {}", path.display()))
    });
    let Err(error) = result else {
        return Ok(());
    };
    match fs::remove_file(&temporary) {
        Ok(()) => Err(error),
        Err(cleanup) if cleanup.kind() == ErrorKind::NotFound => Err(error),
        Err(cleanup) => Err(error.wrap_err(format!(
            "also failed to remove {}: {cleanup}",
            temporary.display()
        ))),
    }
}

// The writer drops, closing the file, before `publish` renames or removes it.
fn write<T: serde::Serialize>(temporary: &Path, report: &T) -> Result<()> {
    let mut writer = std::io::BufWriter::new(
        fs::File::create(temporary).wrap_err_with(|| format!("create {}", temporary.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, report).wrap_err("serialize report")?;
    std::io::Write::flush(&mut writer).wrap_err_with(|| format!("flush {}", temporary.display()))
}
