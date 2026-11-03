/* HOI reconstruction comparison: five synchronized views per object
   (generated video, ours, three baselines) with one shared player. */
(() => {
  'use strict';
  const root = document.getElementById('figure-4');
  const data = window.GALATEA_HOI_DATA;
  if (!root || !data) return;
  const byId = id => document.getElementById(`hoi-${id}`);
  const explorer = byId('explorer');
  const dialog = byId('dialog');
  const playButton = byId('play');
  const timeline = byId('timeline');
  const status = byId('status');
  let index = 0;
  let videos = [];
  let playing = false;
  let generation = 0;
  let playRequest = 0;
  let pendingAutoPlay = false;
  let inView = !('IntersectionObserver' in window);
  let modalResume = false;
  let nearView = !('IntersectionObserver' in window);
  let frame = null;
  const clip = () => data.clips[index];
  const leader = () => videos[1];
  const duration = () => Number.isFinite(leader()?.duration) ? leader().duration : 0;
  const sourceOf = media => media.src + (media.src.endsWith('/input.mp4') ? '?v=20260910-faststart' : '');
  // Start once every view can play through: fully buffered, or HAVE_ENOUGH_DATA
  // (a paused video may stop prefetching before 100%).
  const playable = list => list.length === 5 && list.every(video => !video.error && !video.seeking && video.readyState >= 2 &&
    (video.readyState >= 4 || window.GALATEA_MEDIA.fraction(video) >= .995));

  function setStatus(text, isError = false) {
    status.textContent = text;
    status.classList.toggle('is-error', isError);
  }

  function paintControls() {
    const waiting = pendingAutoPlay && !playing;
    explorer.classList.toggle('is-playing', playing);
    explorer.classList.toggle('is-waiting', waiting);
    playButton.setAttribute('aria-label', playing || waiting ? 'Pause' : 'Play');
  }

  function updateTime() {
    const time = leader()?.currentTime || 0;
    const end = duration() || 5.0417;
    timeline.max = end;
    timeline.value = time;
    timeline.setAttribute('aria-valuetext', `${time.toFixed(1)} of ${end.toFixed(1)} seconds`);
    byId('progress').style.width = `${Math.min(100, time / end * 100)}%`;
  }

  function tick() {
    frame = null;
    updateTime();
    if (playing && !document.hidden) frame = requestAnimationFrame(tick);
  }

  function pause() {
    if (frame !== null) { cancelAnimationFrame(frame); frame = null; }
    pendingAutoPlay = false;
    playRequest += 1;
    playing = false;
    videos.forEach(video => video.pause());
    paintControls();
    updateTime();
  }

  function seek(time) {
    const target = Math.max(0, Math.min(time, Math.max(0, duration() - .001)));
    videos.forEach(video => {
      if (video.readyState >= 1) video.currentTime = Math.min(target, video.duration - .001);
    });
    updateTime();
  }

  async function play() {
    if (!playable(videos)) {
      pendingAutoPlay = true;
      setStatus(window.GALATEA_MEDIA.message(videos));
      paintControls();
      return;
    }
    pendingAutoPlay = false;
    setStatus('');
    const request = ++playRequest;
    const active = videos.slice();
    const version = generation;
    if (leader().currentTime >= duration() - .05) { seek(0); pendingAutoPlay = true; paintControls(); return; }
    playing = true;
    paintControls();
    if (frame === null) frame = requestAnimationFrame(tick);
    const results = await Promise.allSettled(active.map(video => video.play()));
    if (version !== generation) {
      active.forEach(video => video.pause());
      return;
    }
    if (request !== playRequest) {
      if (!playing) active.forEach(video => video.pause());
      return;
    }
    if (results.some(result => result.status === 'rejected')) pause();
  }

  function maybeAutoPlay() {
    if (!pendingAutoPlay || !inView || document.hidden || dialog.open) return;
    videos.forEach(video => { video.preload = 'auto'; });
    if (!playable(videos)) {
      setStatus(window.GALATEA_MEDIA.message(videos));
      paintControls();
      return;
    }
    pendingAutoPlay = false;
    play();
  }

  function suspendPlayback() {
    const resume = pendingAutoPlay || playing;
    pause();
    pendingAutoPlay = resume;
    paintControls();
  }

  function updateStatus(version) {
    if (version !== generation) return;
    const ready = videos.filter(video => video.readyState >= 1 && !video.error).length;
    const failed = videos.some(video => video.error);
    if (failed) pendingAutoPlay = false;
    playButton.disabled = ready !== 5;
    timeline.disabled = ready !== 5;
    if (failed) setStatus('A video could not load. Select the object again to retry.', true);
    else if (ready !== 5) setStatus('Loading');
    updateTime();
    maybeAutoPlay();
  }

  function enlarge(method) {
    const time = leader()?.currentTime || 0;
    pause();
    const source = clip().media[method.id];
    dialog.setAttribute('aria-label', `${clip().label}: ${method.label}`);
    const video = byId('dialog-video');
    video.src = sourceOf(source);
    video.poster = source.poster;
    video.muted = true;
    video.onloadedmetadata = () => {
      if (!dialog.open) return;
      video.currentTime = Math.min(time, video.duration - .001);
      if (!document.hidden) video.play().catch(() => {});
      else modalResume = true;
    };
    dialog.showModal();
  }

  function loadSelectedMedia() {
    videos.forEach(video => {
      if (!video.dataset.src) return;
      video.preload = 'auto';
      video.src = video.dataset.src;
      delete video.dataset.src;
      video.load();
    });
  }

  function select(next) {
    pause();
    if (dialog.open) dialog.close();
    generation += 1;
    const version = generation;
    videos.forEach(video => { video.removeAttribute('src'); video.load(); });
    videos = [];
    index = (next + data.clips.length) % data.clips.length;
    const selected = clip();
    byId('object-select').value = selected.id;
    const nav = byId('objects');
    nav.querySelectorAll('button').forEach((button, i) => button.setAttribute('aria-current', String(i === index)));
    const activeButton = nav.children[index];
    if (activeButton.offsetTop < nav.scrollTop || activeButton.offsetTop + activeButton.offsetHeight > nav.scrollTop + nav.clientHeight) {
      nav.scrollTop = Math.max(0, activeButton.offsetTop - nav.clientHeight / 2);
    }
    byId('videos').replaceChildren();
    data.methods.forEach(method => {
      const media = selected.media[method.id];
      const screen = document.createElement('button');
      screen.type = 'button';
      screen.className = 'hoi-screen';
      screen.dataset.method = method.id;
      screen.setAttribute('aria-label', `Enlarge: ${method.label}`);
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'none';
      video.poster = media.poster;
      video.setAttribute('aria-hidden', 'true');
      ['loadedmetadata', 'loadeddata', 'canplay', 'progress', 'seeked', 'error'].forEach(event =>
        video.addEventListener(event, () => updateStatus(version)));
      video.addEventListener('waiting', () => {
        if (!playing || version !== generation) return;
        pause();
        pendingAutoPlay = true;
        setStatus(window.GALATEA_MEDIA.message(videos));
        paintControls();
      });
      video.addEventListener('ended', () => {
        if (version !== generation || video !== leader()) return;
        const repeat = playing;
        pause();
        if (repeat) { seek(0); pendingAutoPlay = true; paintControls(); maybeAutoPlay(); }
      });
      video.dataset.src = sourceOf(media);
      const tag = document.createElement('span');
      tag.className = method.id === 'ours' ? 'media-tag media-tag--accent' : 'media-tag';
      tag.textContent = method.label;
      tag.setAttribute('aria-hidden', 'true');
      screen.append(video, tag);
      screen.onclick = () => enlarge(method);
      videos.push(video);
      byId('videos').append(screen);
    });
    pendingAutoPlay = true;
    paintControls();
    if (nearView) loadSelectedMedia();
    updateStatus(version);
  }

  data.clips.forEach((item, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'hoi-object';
    button.dataset.object = item.slug;
    button.textContent = item.label;
    button.onclick = () => select(i);
    byId('objects').append(button);
    byId('object-select').add(new Option(item.label, item.id));
  });
  byId('object-select').onchange = event => select(data.clips.findIndex(item => item.id === event.target.value));
  playButton.onclick = () => playing || pendingAutoPlay ? pause() : play();
  byId('prev').onclick = () => select(index - 1);
  byId('next').onclick = () => select(index + 1);
  timeline.oninput = event => { const time = Number(event.target.value); pause(); seek(time); };
  byId('close').onclick = () => dialog.close();
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => {
    modalResume = false;
    const video = byId('dialog-video');
    video.pause();
    video.onloadedmetadata = null;
    video.removeAttribute('src');
    video.load();
    pendingAutoPlay = true;
    maybeAutoPlay();
  });
  root.addEventListener('keydown', event => {
    if (dialog.open || ['INPUT', 'SELECT', 'BUTTON', 'A', 'SUMMARY'].includes(event.target.tagName)) return;
    if (event.code === 'Space') { event.preventDefault(); playing ? pause() : play(); }
    if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
      event.preventDefault();
      pause();
      seek((leader()?.currentTime || 0) + (event.code === 'ArrowRight' ? 1 : -1) / 24);
    }
  });
  document.addEventListener('visibilitychange', () => {
    const video = byId('dialog-video');
    if (document.hidden) {
      suspendPlayback();
      modalResume = dialog.open && (modalResume || !video.paused);
      video.pause();
    } else {
      maybeAutoPlay();
      if (dialog.open && modalResume) { modalResume = false; video.play().catch(() => {}); }
    }
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      nearView = entries[0].isIntersecting;
      if (nearView) loadSelectedMedia();
    }, { rootMargin: '300px 0px' }).observe(explorer);
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      if (inView) maybeAutoPlay(); else suspendPlayback();
    }).observe(explorer);
  }
  setInterval(() => {
    if (!playing || !leader() || leader().seeking) return;
    const time = leader().currentTime;
    videos.forEach(video => {
      if (video !== leader() && video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime - time) > .1) video.currentTime = time;
    });
    updateTime();
  }, 100);
  select(0);
})();
