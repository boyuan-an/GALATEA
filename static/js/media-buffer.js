/* Short synchronized clips are buffered completely before group playback. */
window.GALATEA_MEDIA = {
  fraction(video) {
    if (!Number.isFinite(video.duration) || video.duration <= 0) return 0;
    let end = 0;
    for (let i = 0; i < video.buffered.length; i += 1) {
      if (video.buffered.start(i) > end + .05) break;
      end = Math.max(end, video.buffered.end(i));
    }
    return Math.min(1, end / video.duration);
  },
  ready(videos) {
    return videos.length > 0 && videos.every(video =>
      !video.error && !video.seeking && video.readyState >= 2 && this.fraction(video) >= .995);
  },
  message(videos) {
    const percent = videos.length ? Math.floor(Math.min(...videos.map(video => this.fraction(video))) * 100) : 0;
    return `Buffering HD videos… ${percent}%`;
  }
};
