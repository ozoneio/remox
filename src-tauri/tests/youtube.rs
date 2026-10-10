use remox_desktop_lib::youtube::{validate_url, YoutubeError};

#[test]
fn video_links_canonicalize_to_one_video_without_playlist_parameters() {
    for input in [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLexample&index=2",
        "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ?si=tracking",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=1#fragment",
    ] {
        let video = validate_url(input).expect("valid single-video link");
        assert_eq!(video.id, "dQw4w9WgXcQ");
        assert_eq!(video.url, "https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    }
}

#[test]
fn malformed_or_non_video_urls_fail_before_network_access() {
    for input in [
        "http://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.example.com/watch?v=dQw4w9WgXcQ",
        "https://127.0.0.1/watch?v=dQw4w9WgXcQ",
        "https://localhost/watch?v=dQw4w9WgXcQ",
        "file:///C:/music.wav",
        "https://user@youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com:443/watch?v=dQw4w9WgXcQ",
        "https://youtube.com:8443/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/playlist?list=PLexample",
        "https://youtube.com/watch?list=PLexample",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&v=dQw4w9WgXcQ",
        "https://youtube.com/embed/dQw4w9WgXcQ",
        "https://youtube.com/watch?v=tooShort",
        "https://youtu.be/dQw4w9WgXcQ/extra",
        "https://youtu.be/dQw4w9WgXcQ%2F",
        "https://youtube.com/shorts/",
        "https://youtube.com/watch?v=dQw4w9WgXc!",
        "https://youtube.com/watch?v=dQw4w9WgX\ncQ",
        "https://youtube.com\\watch?v=dQw4w9WgXcQ",
    ] {
        assert!(
            matches!(validate_url(input), Err(YoutubeError::InvalidUrl { .. })),
            "accepted invalid URL: {input}"
        );
    }
}
