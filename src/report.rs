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

pub fn publish(directory: &Path, file: &str, report: &serde_json::Value) -> Result<()> {
    let path = directory.join(file);
    let temporary = directory.join(format!("{file}.tmp"));
    let bytes = serde_json::to_vec_pretty(report).wrap_err("serialize report")?;
    fs::write(&temporary, bytes).wrap_err_with(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, &path).wrap_err_with(|| format!("publish {}", path.display()))
}
