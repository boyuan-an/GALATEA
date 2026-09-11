/* Scoped controls for the static, portable selection of comparison examples. */
(() => {
  'use strict';
  const root = document.getElementById('figure-4');
  const data = window.GALATEA_HOI_DATA;
  if (!root || !data) return;
  const byId = id => document.getElementById(`hoi-${id}`);
  const dialog = byId('dialog');
  let index = 0;
  let videos = [];
  let playing = false;
  let generation = 0;
  let playRequest = 0;
  let speed = 1;
  let pendingAutoPlay = false;
  let inView = !('IntersectionObserver' in window);
  let modalResume = false;
  let nearView = !('IntersectionObserver' in window);
  const clip = () => data.clips[index];
  const leader = () => videos[1];
  const duration = () => Number.isFinite(leader()?.duration) ? leader().duration : 0;

  function updateTime() {
    const time = leader()?.currentTime || 0;
    const end = duration();
    byId('timeline').max = end || 5.0417;
    byId('timeline').value = time;
    byId('time').textContent = `${time.toFixed(2)} / ${(end || 5.0417).toFixed(2)} s`;
  }

  function pause() {
    if (pendingAutoPlay) byId('status').textContent = 'Paused';
    pendingAutoPlay = false;
    playRequest += 1;
    playing = false;
    videos.forEach(video => video.pause());
    byId('play').textContent = 'Play all';
    byId('play').setAttribute('aria-label', 'Play all videos');
    updateTime();
  }

  function seek(time) {
    const target = Math.max(0, Math.min(time, Math.max(0, duration() - .001)));
    videos.forEach(video => {
      if (video.readyState >= 1) video.currentTime = Math.min(target, video.duration - .001);
    });
    byId('status').textContent = videos.some(video => video.seeking) ? 'Loading selected frames…' : 'Synchronized playback';
    updateTime();
  }

  async function play() {
    if (!window.GALATEA_MEDIA.ready(videos)) {
      pendingAutoPlay = true;
      byId('status').textContent = window.GALATEA_MEDIA.message(videos);
      byId('play').textContent = 'Cancel autoplay';
      byId('play').setAttribute('aria-label', 'Cancel autoplay');
      return;
    }
    pendingAutoPlay = false;
    if (videos.length !== 5 || videos.some(video => video.readyState < 1 || video.error)) return;
    const request = ++playRequest;
    const active = videos.slice();
    const version = generation;
    if (leader().currentTime >= duration() - .05) { seek(0); pendingAutoPlay = true; return; }
    playing = true;
    byId('play').textContent = 'Pause all';
    byId('play').setAttribute('aria-label', 'Pause all videos');
    const results = await Promise.allSettled(active.map(video => video.play()));
    if (version !== generation) {
      active.forEach(video => video.pause());
      return;
    }
    if (request !== playRequest) {
      if (!playing) active.forEach(video => video.pause());
      return;
    }
    if (results.some(result => result.status === 'rejected')) {
      pause();
      byId('status').textContent = 'Playback was interrupted. Select Play all to try again.';
    }
  }

  function maybeAutoPlay() {
    if (!pendingAutoPlay || !inView || document.hidden || dialog.open) return;
    videos.forEach(video => { video.preload = 'auto'; });
    if (!window.GALATEA_MEDIA.ready(videos)) {
      byId('status').textContent = window.GALATEA_MEDIA.message(videos);
      byId('play').textContent = 'Cancel autoplay';
      byId('play').setAttribute('aria-label', 'Cancel autoplay');
      return;
    }
    if (videos.length !== 5 || videos.some(video => video.readyState < 2 || video.seeking || video.error)) return;
    pendingAutoPlay = false;
    play();
  }

  function suspendPlayback() {
    const resume = pendingAutoPlay || playing;
    pause();
    pendingAutoPlay = resume;
  }

  function updateStatus(version) {
    if (version !== generation) return;
    const ready = videos.filter(video => video.readyState >= 1 && !video.error).length;
    const failed = videos.some(video => video.error);
    if (failed) pendingAutoPlay = false;
    byId('play').disabled = ready !== 5;
    byId('timeline').disabled = ready !== 5;
    byId('status').textContent = failed ? 'A video could not load. Select the object again to retry.' :
      ready === 5 ? 'Synchronized playback' : 'Loading videos…';
    updateTime();
    maybeAutoPlay();
  }

  function enlarge(method) {
    const time = leader()?.currentTime || 0;
    pause();
    const source = clip().media[method.id];
    byId('dialog-title').textContent = `${clip().label} / ${method.label}`;
    const video = byId('dialog-video');
    video.src = source.src + (source.src.endsWith('/input.mp4') ? '?v=20260910-faststart' : '');
    video.poster = source.poster;
    video.muted = true;
    video.playbackRate = speed;
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
    byId('object-name').textContent = selected.label;
    byId('action').textContent = selected.action;
    byId('object-select').value = selected.id;
    byId('objects').querySelectorAll('button').forEach((button, i) => {
      button.setAttribute('aria-current', String(i === index));
    });
    const activeButton = byId('objects').children[index];
    const nav = byId('objects');
    if (activeButton.offsetTop < nav.scrollTop || activeButton.offsetTop + activeButton.offsetHeight > nav.scrollTop + nav.clientHeight) {
      nav.scrollTop = Math.max(0, activeButton.offsetTop - nav.offsetTop - nav.clientHeight / 2);
    }
    byId('videos').replaceChildren();
    data.methods.forEach(method => {
      const media = selected.media[method.id];
      const card = document.createElement('article');
      card.className = 'hoi-card';
      card.dataset.method = method.id;
      const heading = document.createElement('div');
      heading.className = 'hoi-card-heading';
      const label = document.createElement('h4');
      label.className = 'hoi-method';
      label.textContent = method.label;
      heading.append(label);
      const screen = document.createElement('button');
      screen.type = 'button';
      screen.className = 'hoi-screen';
      screen.setAttribute('aria-label', `Enlarge ${method.label}`);
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'none';
      video.playbackRate = speed;
      video.poster = media.poster;
      video.setAttribute('aria-label', `${selected.label} — ${method.label}`);
      ['loadedmetadata', 'loadeddata', 'canplay', 'progress', 'seeked', 'error'].forEach(event =>
        video.addEventListener(event, () => updateStatus(version)));
      video.addEventListener('waiting', () => {
        if (!playing || version !== generation) return;
        pause();
        pendingAutoPlay = true;
        byId('status').textContent = window.GALATEA_MEDIA.message(videos);
      });
      video.addEventListener('ended', () => {
        if (version !== generation || video !== leader()) return;
        const repeat = playing && byId('loop').checked;
        pause();
        if (repeat) { seek(0); pendingAutoPlay = true; maybeAutoPlay(); }
      });
      video.dataset.src = media.src + (media.src.endsWith('/input.mp4') ? '?v=20260910-faststart' : '');
      const expand = document.createElement('span');
      expand.className = 'hoi-expand';
      expand.textContent = 'Enlarge ↗';
      expand.setAttribute('aria-hidden', 'true');
      screen.append(video, expand);
      screen.onclick = () => enlarge(method);
      card.append(heading, screen);
      videos.push(video);
      byId('videos').append(card);
    });
    pendingAutoPlay = true;
    if (nearView) loadSelectedMedia();
    updateStatus(version);
  }

  data.clips.forEach((item, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'hoi-object';
    button.dataset.object = item.slug;
    const label = document.createElement('span');
    label.textContent = item.label;
    button.append(label);
    button.onclick = () => select(i);
    byId('objects').append(button);
    byId('object-select').add(new Option(item.label, item.id));
  });
  byId('object-select').onchange = event => select(data.clips.findIndex(item => item.id === event.target.value));
  byId('play').onclick = () => playing || pendingAutoPlay ? pause() : play();
  byId('prev').onclick = () => select(index - 1);
  byId('next').onclick = () => select(index + 1);
  byId('restart').onclick = () => seek(0);
  byId('timeline').oninput = event => { const time = Number(event.target.value); pause(); seek(time); };
  byId('speed').onchange = event => { speed = Number(event.target.value); videos.forEach(video => { video.playbackRate = speed; }); };
  byId('layout').onclick = () => {
    const wide = byId('explorer').classList.toggle('is-five-column');
    byId('layout').setAttribute('aria-pressed', String(wide));
    byId('layout').textContent = wide ? 'Larger views' : 'Side-by-side view';
  };
  byId('close').onclick = () => dialog.close();
  dialog.addEventListener('close', () => {
    modalResume = false;
    const video = byId('dialog-video');
    video.pause();
    video.onloadedmetadata = null;
    video.removeAttribute('src');
    video.load();
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
    }, { rootMargin: '300px 0px' }).observe(byId('explorer'));
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      if (inView) maybeAutoPlay(); else suspendPlayback();
    }).observe(byId('explorer'));
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
