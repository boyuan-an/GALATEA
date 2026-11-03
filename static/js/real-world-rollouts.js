/* Real-world rollouts: a three-row filmstrip that scrolls sideways, with a drag bar and
   arrow buttons. Only clips on screen load and play; a click opens the clip with controls. */
(() => {
  'use strict';
  const strip = document.getElementById('rollout-grid');
  const nav = document.getElementById('rollout-nav');
  const bar = document.getElementById('rollout-scrollbar');
  const thumb = document.getElementById('rollout-thumb');
  const prev = document.getElementById('rollout-prev');
  const next = document.getElementById('rollout-next');
  const dialog = document.getElementById('rollout-dialog');
  const items = window.GALATEA_ROLLOUTS;
  if (!strip || !Array.isArray(items) || !items.length) return;
  const longHorizon = items.findIndex(item => item.group === 'Long-horizon');
  const featured = new Set([0, longHorizon].filter(index => index >= 0));
  const path = (item, ext) => `./videos/real-world-rollouts/rollout-${item.id}.${ext}`;

  strip.innerHTML = items.map((item, index) => `
    <button type="button" class="rollout-tile${featured.has(index) ? ' rollout-tile--featured' : ''}"
      data-index="${index}" aria-label="Play rollout ${item.id}">
      <video muted loop playsinline preload="none" disablepictureinpicture aria-hidden="true"
        data-poster="${path(item, 'jpg')}" data-src="${path(item, 'mp4')}" data-bytes="${item.bytes || 0}"></video>
    </button>`).join('');

  const tiles = [...strip.querySelectorAll('.rollout-tile')];
  const videos = tiles.map(tile => tile.querySelector('video'));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const saveData = Boolean(navigator.connection?.saveData);
  let limit = reducedMotion || saveData ? 0 : 3;
  let sectionVisible = false;
  const onScreen = new Set();
  const startedAt = new WeakMap();
  const measured = new WeakSet();

  const showPoster = video => {
    if (video.dataset.poster) { video.poster = video.dataset.poster; delete video.dataset.poster; }
  };

  function load(video) {
    showPoster(video);
    if (!video.dataset.src) return;
    startedAt.set(video, performance.now());
    video.preload = 'auto';
    video.src = video.dataset.src;
    delete video.dataset.src;
    video.load();
  }

  function release(video) {
    video.pause();
    // Abort partial downloads of clips that left the screen; keep completed ones cached.
    if (video.getAttribute('src') && !window.GALATEA_MEDIA.ready([video])) {
      video.dataset.src = video.getAttribute('src');
      video.removeAttribute('src');
      video.preload = 'none';
      video.load();
    }
  }

  function refresh() {
    const active = sectionVisible && !document.hidden && !dialog?.open;
    const chosen = active ? videos.filter(video => onScreen.has(video)).slice(0, limit) : [];
    for (const video of videos) {
      if (!chosen.includes(video)) { release(video); continue; }
      load(video);
      // Play once the clip can play through; a paused clip may stop prefetching before 100%.
      if (video.error || video.readyState < 4 && !window.GALATEA_MEDIA.ready([video])) continue;
      if (!measured.has(video) && startedAt.has(video)) {
        measured.add(video);
        const seconds = Math.max(.05, (performance.now() - startedAt.get(video)) / 1000);
        const mbps = Number(video.dataset.bytes) * 8 / seconds / 1e6;
        // Start with a few clips; allow more only after a fast complete transfer.
        if (mbps >= 8 && limit < 10) { limit = 10; queueMicrotask(refresh); }
      }
      if (video.paused) video.play().catch(() => {});
    }
  }

  videos.forEach(video => {
    for (const event of ['loadeddata', 'progress', 'canplay', 'seeked']) video.addEventListener(event, refresh);
    video.addEventListener('error', () => {
      if (video.getAttribute('src')) { video.dataset.src = video.getAttribute('src'); video.removeAttribute('src'); }
    });
  });

  /* Drag bar: the thumb mirrors the visible share of the strip and drags it. */
  let drag = null;
  function paintBar() {
    if (!bar || !thumb) return;
    const max = strip.scrollWidth - strip.clientWidth;
    const track = bar.clientWidth;
    const width = Math.max(48, track * strip.clientWidth / Math.max(1, strip.scrollWidth));
    thumb.style.width = `${width}px`;
    thumb.style.transform = `translateX(${max > 0 ? (track - width) * strip.scrollLeft / max : 0}px)`;
    if (nav) nav.hidden = max <= 1;
    if (prev) prev.disabled = strip.scrollLeft <= 1;
    if (next) next.disabled = strip.scrollLeft >= max - 1;
  }
  let painting = false;
  const schedulePaint = () => {
    if (painting) return;
    painting = true;
    requestAnimationFrame(() => { painting = false; paintBar(); });
  };
  if (bar && thumb) {
    thumb.addEventListener('pointerdown', event => {
      drag = { x: event.clientX, left: strip.scrollLeft };
      thumb.setPointerCapture(event.pointerId);
      strip.classList.add('is-dragging');
      event.preventDefault();
    });
    thumb.addEventListener('pointermove', event => {
      if (!drag) return;
      const travel = bar.clientWidth - thumb.offsetWidth;
      const max = strip.scrollWidth - strip.clientWidth;
      strip.scrollLeft = drag.left + (event.clientX - drag.x) * (travel > 0 ? max / travel : 0);
    });
    // Snapping is off while dragging; on release, settle on the nearest whole column.
    const endDrag = () => {
      if (!drag) return;
      drag = null;
      strip.classList.remove('is-dragging');
      const small = tiles.find(tile => !tile.classList.contains('rollout-tile--featured'));
      const gap = parseFloat(getComputedStyle(strip).columnGap) || 0;
      const pitch = small ? small.offsetWidth + gap : 0;
      if (pitch > 0) strip.scrollTo({ left: Math.round(strip.scrollLeft / pitch) * pitch, behavior: 'smooth' });
    };
    thumb.addEventListener('pointerup', endDrag);
    thumb.addEventListener('pointercancel', endDrag);
    bar.addEventListener('pointerdown', event => {
      if (event.target === thumb) return;
      const rect = bar.getBoundingClientRect();
      const ratio = (event.clientX - rect.left - thumb.offsetWidth / 2) / Math.max(1, rect.width - thumb.offsetWidth);
      strip.scrollTo({ left: Math.max(0, Math.min(1, ratio)) * (strip.scrollWidth - strip.clientWidth), behavior: 'smooth' });
    });
  }
  const page = direction => {
    const gap = parseFloat(getComputedStyle(strip).columnGap) || 0;
    strip.scrollBy({ left: direction * (strip.clientWidth + gap), behavior: 'smooth' });
  };
  if (prev) prev.onclick = () => page(-1);
  if (next) next.onclick = () => page(1);
  strip.addEventListener('scroll', schedulePaint, { passive: true });
  window.addEventListener('resize', schedulePaint);
  if ('ResizeObserver' in window) new ResizeObserver(schedulePaint).observe(strip);
  paintBar();

  if (dialog) {
    const player = dialog.querySelector('video');
    dialog.querySelector('.media-dialog__close').onclick = () => dialog.close();
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => {
      player.pause();
      player.removeAttribute('src');
      player.load();
      refresh();
    });
    tiles.forEach((tile, index) => tile.onclick = () => {
      const item = items[index];
      dialog.setAttribute('aria-label', `Rollout ${item.id}`);
      player.poster = path(item, 'jpg');
      player.src = path(item, 'mp4');
      dialog.showModal();
      refresh();
      player.play().catch(() => {});
    });
  }

  if (!('IntersectionObserver' in window)) {
    sectionVisible = true;
    videos.slice(0, 8).forEach(video => { showPoster(video); onScreen.add(video); });
    refresh();
    return;
  }
  // Intersection accounts for the strip's horizontal clipping, so off-strip clips count as hidden.
  const posterObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) { showPoster(entry.target.querySelector('video')); posterObserver.unobserve(entry.target); } });
  }, { rootMargin: '400px' });
  const playObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const video = entry.target.querySelector('video');
      if (entry.intersectionRatio >= .5) onScreen.add(video);
      else onScreen.delete(video);
    });
    refresh();
  }, { threshold: [0, .5] });
  tiles.forEach(tile => { posterObserver.observe(tile); playObserver.observe(tile); });
  new IntersectionObserver(entries => {
    sectionVisible = entries[0].isIntersecting;
    refresh();
  }, { threshold: 0 }).observe(strip);
  document.addEventListener('visibilitychange', refresh);
})();
