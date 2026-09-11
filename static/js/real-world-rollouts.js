(() => {
  'use strict';
  const viewport = document.querySelector('.rollout-reel__viewport');
  const grid = document.getElementById('rollout-grid');
  const items = window.GALATEA_ROLLOUTS;
  if (!viewport || !grid || !Array.isArray(items) || !items.length) return;
  grid.innerHTML = items.map(item => `
    <article class="rollout-scene" data-rollout-scene>
      <figure class="rollout-scene__media">
        <video controls controlslist="nodownload" muted loop playsinline preload="none"
          data-poster="./videos/real-world-rollouts/rollout-${item.id}.jpg"
          data-src="./videos/real-world-rollouts/rollout-${item.id}.mp4"
          data-bytes="${item.bytes || 0}" aria-label="${item.group} physical rollout ${item.id}"></video>
        <button class="rollout-load-button" type="button" aria-label="Play rollout ${item.id}">▶ Play</button>
        <span class="rollout-buffer-status" role="status"></span>
      </figure>
    </article>`).join('');
  const scenes = [...grid.querySelectorAll('[data-rollout-scene]')];
  const videos = scenes.map(scene => scene.querySelector('video'));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const connection = navigator.connection;
  let limit = 1;
  let visible = false;
  let initialized = false;
  let manual = null;
  const candidates = new Set();
  const started = new WeakMap();
  const measured = new WeakSet();
  const approved = new Set();
  const showPoster = video => {
    if (video.dataset.poster) { video.poster = video.dataset.poster; delete video.dataset.poster; }
  };
  function load(video) {
    showPoster(video);
    if (!video.dataset.src) return;
    started.set(video, performance.now());
    video.preload = 'auto';
    video.src = video.dataset.src;
    delete video.dataset.src;
    video.load();
  }
  function refresh() {
    const onScreen = videos.filter(video => candidates.has(video));
    const chosen = visible && !document.hidden ?
      (manual && candidates.has(manual) ? [manual] : reducedMotion ? [] : onScreen.slice(0, limit)) : [];
    for (const video of videos) {
      const selected = chosen.includes(video);
      const screen = video.parentElement;
      const status = screen.querySelector('.rollout-buffer-status');
      const button = screen.querySelector('.rollout-load-button');
      if (!selected) {
        video.pause();
        approved.delete(video);
        status.textContent = '';
        button.hidden = false;
        // Abort pending off-screen or superseded downloads; keep completed clips cached.
        if (video.getAttribute('src') && !window.GALATEA_MEDIA.ready([video])) {
          video.dataset.src = video.getAttribute('src');
          video.removeAttribute('src');
          video.preload = 'none';
          video.load();
        }
        continue;
      }
      load(video);
      button.hidden = true;
      if (!window.GALATEA_MEDIA.ready([video])) {
        status.textContent = window.GALATEA_MEDIA.message([video]);
        continue;
      }
      status.textContent = '';
      if (!measured.has(video) && started.has(video)) {
        measured.add(video);
        const seconds = Math.max(.05, (performance.now() - started.get(video)) / 1000);
        const mbps = Number(video.dataset.bytes) * 8 / seconds / 1e6;
        // Start conservatively; expand only after a real complete-video transfer.
        if (!manual && !connection?.saveData && mbps >= 8) {
          limit = 9;
          queueMicrotask(refresh);
        }
      }
      if (!approved.has(video)) {
        approved.add(video);
        video.play().catch(() => { approved.delete(video); button.hidden = false; });
      }
    }
  }
  videos.forEach(video => {
    video.parentElement.querySelector('button').onclick = () => { manual = video; refresh(); };
    for (const event of ['loadeddata', 'progress', 'canplay', 'seeked']) video.addEventListener(event, refresh);
    video.addEventListener('error', () => {
      approved.delete(video);
      video.parentElement.querySelector('.rollout-buffer-status').textContent = 'Could not load. Select Play to retry.';
      video.parentElement.querySelector('button').hidden = false;
      if (video.getAttribute('src')) video.dataset.src = video.getAttribute('src');
    });
  });
  if (!('IntersectionObserver' in window)) {
    visible = true;
    videos.slice(0, 9).forEach(video => candidates.add(video));
    refresh();
    return;
  }
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const video = entry.target.querySelector('video');
      if (entry.isIntersecting) showPoster(video);
      if (entry.intersectionRatio >= .55) candidates.add(video);
      else { candidates.delete(video); if (manual === video) manual = null; }
    });
    refresh();
  }, { root: viewport, threshold: [0, .55] });
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (visible && !initialized) {
      initialized = true;
      scenes.forEach(scene => observer.observe(scene));
    }
    refresh();
  }, { threshold: .01 }).observe(viewport);
  document.addEventListener('visibilitychange', refresh);
})();
