pub mod engine;
pub mod media;
pub mod youtube;

#[cfg(feature = "desktop")]
mod commands;

#[cfg(feature = "desktop")]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(commands::DesktopSession::new())
        .invoke_handler(tauri::generate_handler![
            commands::import_audio,
            commands::import_youtube,
            commands::generate_remix,
            commands::export_remix
        ])
        .run(tauri::generate_context!())
        .expect("Failed to start Remox: check WebView2 and the desktop runtime installation");
}
