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

pub fn publish<T: serde::Serialize>(directory: &Path, file: &str, report: &T) -> Result<()> {
    let path = directory.join(file);
    let temporary = directory.join(format!("{file}.tmp"));
    let mut writer = std::io::BufWriter::new(
        fs::File::create(&temporary).wrap_err_with(|| format!("create {}", temporary.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, report).wrap_err("serialize report")?;
    std::io::Write::flush(&mut writer)
        .wrap_err_with(|| format!("flush {}", temporary.display()))?;
    drop(writer);
    fs::rename(&temporary, &path).wrap_err_with(|| format!("publish {}", path.display()))
}
