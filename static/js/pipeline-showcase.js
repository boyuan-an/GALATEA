/* Highlights: one reference clock drives the generated video, three reconstruction
   views cropped from the same stream, and a keyframe-synchronized real-world recording. */
(() => {
  'use strict';
  const root = document.getElementById('pipeline-showcase');
  const data = window.GALATEA_PIPELINE_DATA;
  if (!root || !data?.samples?.length) return;
  const byId = (id) => root.querySelector(`#ps-${id}`);
  const video = byId('video');
  const execution = byId('execution-video');
  const dialog = byId('dialog');
  const modalVideo = byId('dialog-video');
  const playButton = byId('play');
  const panel = byId('panel');
  const scrub = byId('scrub');
  const progress = byId('progress');
  const status = byId('status');
  const stages = [
    { id: 'input', label: 'Generated video' },
    { id: 'depth', label: 'Depth (MoGe-2)' },
    { id: 'sam2', label: 'Object mask (SAM 2)' },
    { id: 'trajectory', label: 'Reconstructed hand and object' }
  ];
  const executionStage = { id: 'execution', label: 'Real-world execution' };
  const views = [];
  let selected = data.samples[0];
  let generation = 0;
  let playRequest = 0;
  let animation = null;
  let displayedTime = null;
  let requestedTime = 0;
  let pendingSeek = false;
  let inView = false;
  let nearView = !('IntersectionObserver' in window);
  let pendingAutoPlay = false;
  let modalFrameCallback = null;
  let issuedTarget = null;

  // Start once both streams can play through: fully buffered, or the browser reports
  // HAVE_ENOUGH_DATA (it may stop prefetching a paused video before 100%).
  const playable = (videos) => videos.every((media) => !media.error && !media.seeking && media.readyState >= 2 &&
    (media.readyState >= 4 || window.GALATEA_MEDIA.fraction(media) >= .995));

  const duration = () => selected.frames / selected.fps;
  const url = (path) => `${path}?v=${data.mediaVersion}`;
  const frameTarget = (sample, time) => Math.min(Math.max(0, time), (sample.frames - 1) / sample.fps) + .0001;
  const keyframes = () => selected.syncKeyframes || [[0, 0], [duration(), selected.execution.duration]];

  function mapExecutionTime(referenceTime) {
    const points = keyframes();
    const t = Math.max(0, Math.min(referenceTime, points.at(-1)[0]));
    for (let i = 1; i < points.length; i += 1) {
      const [r1, e1] = points[i];
      const [r0, e0] = points[i - 1];
      if (t <= r1) {
        const ratio = (t - r0) / Math.max(.0001, r1 - r0);
        return e0 + ratio * (e1 - e0);
      }
    }
    return points.at(-1)[1];
  }

  function executionRate(referenceTime) {
    const points = keyframes();
    for (let i = 1; i < points.length; i += 1) {
      if (referenceTime <= points[i][0]) {
        return Math.max(.1, (points[i][1] - points[i - 1][1]) / Math.max(.0001, points[i][0] - points[i - 1][0]));
      }
    }
    return 1;
  }

  function phaseAt(sample, time) {
    const frame = Math.max(0, Math.min(sample.frames - 1, Math.floor(time * sample.fps + 1e-4)));
    if (sample.contact.some(([a, b]) => frame >= a && frame <= b)) return 'contact';
    return frame < sample.contact[0][0] ? 'pre' : frame > sample.contact.at(-1)[1] ? 'post' : 'gap';
  }

  function setStatus(text, isError = false) {
    status.textContent = text;
    status.classList.toggle('is-error', isError);
  }

  function syncExecution(referenceTime, force = false) {
    if (!execution || execution.readyState < 1) return;
    const target = mapExecutionTime(referenceTime);
    execution.playbackRate = executionRate(referenceTime);
    const drift = Math.abs(execution.currentTime - target);
    if ((force && drift > .015) || (!execution.seeking && drift > .2)) execution.currentTime = target;
  }

  function drawViews() {
    if (video.seeking || video.readyState < 2 || !video.videoWidth) return;
    for (const view of views) {
      const [x, y, width, height] = data.frameLayout.regions[view.stage];
      if (view.canvas.width !== width || view.canvas.height !== height) {
        view.canvas.width = width;
        view.canvas.height = height;
      }
      view.context.drawImage(video, x, y, width, height, 0, 0, width, height);
      view.canvas.dataset.frame = String(Math.floor((displayedTime ?? video.currentTime) * selected.fps + 1e-4));
    }
  }

  function paint() {
    const time = Math.min(pendingSeek ? requestedTime : video.readyState >= 1 ? video.currentTime : requestedTime, duration());
    panel.dataset.phase = root.dataset.phase = phaseAt(selected, displayedTime ?? time);
    scrub.value = time;
    scrub.setAttribute('aria-valuetext', `${time.toFixed(1)} of ${duration().toFixed(1)} seconds`);
    progress.style.width = `${time / duration() * 100}%`;
    if (!pendingSeek) syncExecution(time);
  }

  function animate() {
    animation = null;
    displayedTime = video.currentTime;
    drawViews();
    paint();
    if (!video.paused && !document.hidden) animation = requestAnimationFrame(animate);
  }

  function paintControls() {
    const ready = video.readyState >= 1 && execution.readyState >= 1 && !video.error && !execution.error;
    const playing = !video.paused;
    const waiting = pendingAutoPlay && video.paused;
    playButton.disabled = !ready && !waiting;
    root.classList.toggle('is-playing', playing);
    root.classList.toggle('is-waiting', waiting);
    playButton.setAttribute('aria-label', playing || waiting ? 'Pause' : 'Play');
  }

  function pauseBoth() {
    pendingAutoPlay = false;
    playRequest += 1;
    video.pause();
    execution.pause();
    if (animation !== null) { cancelAnimationFrame(animation); animation = null; }
    paintControls();
    paint();
  }

  async function playBoth() {
    if (video.error || execution.error) return;
    pendingAutoPlay = true;
    video.preload = execution.preload = 'auto';
    if (pendingSeek || !playable([video, execution])) {
      setStatus(window.GALATEA_MEDIA.message([video, execution]));
      paintControls();
      return;
    }
    pendingAutoPlay = false;
    setStatus('');
    const request = ++playRequest;
    if (video.ended || video.currentTime >= duration() - .02) {
      video.currentTime = 0;
      execution.currentTime = 0;
    }
    syncExecution(video.currentTime, true);
    try {
      await Promise.all([video.play(), execution.play()]);
      if (request !== playRequest) return;
      if (animation === null) animation = requestAnimationFrame(animate);
      paintControls();
    } catch {
      if (request !== playRequest) return;
      pauseBoth();
    }
  }

  function seek(time) {
    requestedTime = Math.max(0, Math.min(time, duration()));
    pendingSeek = true;
    seekReference();
    paint();
  }

  function seekReference() {
    if (!pendingSeek || video.readyState < 1 || video.seeking) return;
    const target = frameTarget(selected, requestedTime);
    // A seek counts as done once the browser finished the one we issued for this target;
    // the reported currentTime can differ from the request by rounding, so never re-seek on that.
    const finished = issuedTarget !== null && Math.abs(issuedTarget - target) < 1e-6;
    if (finished || Math.abs(video.currentTime - target) < .5 / selected.fps) {
      issuedTarget = null;
      pendingSeek = false;
      displayedTime = video.currentTime;
      drawViews();
      paint();
      return;
    }
    // Let decoding finish before applying the latest drag position.
    issuedTarget = target;
    video.currentTime = target;
  }

  function maybeAutoPlay() {
    if (pendingSeek || !pendingAutoPlay || !inView || document.hidden || dialog.open) return;
    if (video.readyState < 1 || execution.readyState < 1) return;
    playBoth();
  }

  function timeline() {
    const spans = [];
    let end = 0;
    selected.contact.forEach(([a, b], i) => {
      const start = a / selected.fps;
      const stop = (b + 1) / selected.fps;
      if (start > end) spans.push([end, start, i ? 'gap' : 'pre']);
      spans.push([start, stop, 'contact']);
      end = stop;
    });
    if (end < duration()) spans.push([end, duration(), 'post']);
    const place = (a, b) => `left:${a / duration() * 100}%;width:${(b - a) / duration() * 100}%`;
    byId('segments').innerHTML = spans.map(([a, b, phase]) => `<span class="player-segment" data-phase="${phase}" style="${place(a, b)}"></span>`).join('') +
      spans.filter(([, , phase]) => phase === 'contact').map(([a, b]) => `<span class="player-segment-label" style="${place(a, b)}"><span class="label-long">Auto-detected contact window (HOI-DETR)</span><span class="label-short">Contact (HOI-DETR)</span></span>`).join('');
    scrub.max = duration();
  }

  function loadSelectedMedia() {
    for (const media of [video, execution]) {
      if (!media.dataset.src) continue;
      media.preload = 'auto';
      media.src = media.dataset.src;
      delete media.dataset.src;
      media.load();
    }
  }

  function select(sample) {
    pauseBoth();
    if (dialog.open) dialog.close();
    generation += 1;
    selected = sample;
    pendingSeek = false;
    issuedTarget = null;
    displayedTime = null;
    requestedTime = 0;
    panel.dataset.sample = selected.id;
    byId('samples').querySelectorAll('button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.sample === selected.id)));
    setStatus('Loading');
    scrub.disabled = true;
    for (const view of views) {
      view.context.clearRect(0, 0, view.canvas.width, view.canvas.height);
      delete view.canvas.dataset.frame;
      view.poster.src = url(selected.media[view.stage].poster);
    }
    timeline();
    video.poster = url(selected.media.pipeline.poster);
    execution.poster = url(selected.execution.poster);
    video.removeAttribute('src');
    execution.removeAttribute('src');
    video.dataset.src = url(selected.media.pipeline.src);
    execution.dataset.src = url(selected.execution.src);
    video.loop = false;
    video.playbackRate = 1;
    execution.playbackRate = executionRate(0);
    video.preload = execution.preload = 'none';
    video.load();
    execution.load();
    if (nearView) loadSelectedMedia();
    pendingAutoPlay = true;
    paint();
    paintControls();
  }

  function enlarge(stage) {
    const time = video.currentTime;
    pauseBoth();
    dialog.setAttribute('aria-label', `${selected.label}: ${stage.label}`);
    const source = stage.id === 'execution' ? selected.execution : selected.media[stage.id];
    modalVideo.poster = url(source.poster);
    modalVideo.onloadedmetadata = () => {
      if (!dialog.open) return;
      modalVideo.currentTime = stage.id === 'execution' ? mapExecutionTime(time) : frameTarget(selected, time);
      modalVideo.playbackRate = stage.id === 'execution' ? executionRate(time) : 1;
      modalVideo.play().catch(() => {});
    };
    modalVideo.src = url(source.src);
    dialog.dataset.phase = phaseAt(selected, time);
    dialog.showModal();
    if (modalVideo.requestVideoFrameCallback) {
      const version = generation;
      const update = (_, meta) => {
        if (!dialog.open || version !== generation) return;
        dialog.dataset.phase = phaseAt(selected, meta.mediaTime);
        modalFrameCallback = modalVideo.requestVideoFrameCallback(update);
      };
      modalFrameCallback = modalVideo.requestVideoFrameCallback(update);
    }
  }

  byId('samples').innerHTML = data.samples.map((sample) => `<button type="button" class="ps-sample" data-sample="${sample.id}" aria-pressed="false" aria-label="${sample.label}" title="${sample.label}"><img src="${url(sample.execution.poster)}" alt="" loading="lazy" decoding="async"></button>`).join('');
  byId('samples').querySelectorAll('button').forEach((button) => button.onclick = () => {
    if (button.dataset.sample !== selected.id) select(data.samples.find((sample) => sample.id === button.dataset.sample));
  });
  byId('stages').innerHTML = stages.slice(1).map((stage) => `<button type="button" class="ps-recon__tile" data-stage="${stage.id}" aria-label="Enlarge: ${stage.label}" title="${stage.label}"><img alt="" aria-hidden="true"><canvas width="424" height="240" aria-hidden="true"></canvas></button>`).join('');
  byId('stages').querySelectorAll('button').forEach((button) => {
    const canvas = button.querySelector('canvas');
    views.push({ stage: button.dataset.stage, canvas, context: canvas.getContext('2d'), poster: button.querySelector('img') });
  });
  root.querySelectorAll('[data-stage]').forEach((button) => button.onclick = () => enlarge(stages.find((stage) => stage.id === button.dataset.stage) || executionStage));

  video.addEventListener('loadedmetadata', () => { video.currentTime = frameTarget(selected, requestedTime); video.playbackRate = 1; syncExecution(requestedTime, true); paint(); });
  video.addEventListener('loadeddata', () => { setStatus(''); scrub.disabled = false; drawViews(); paint(); paintControls(); maybeAutoPlay(); });
  video.addEventListener('timeupdate', () => { if (pendingSeek || video.seeking) return; displayedTime = video.currentTime; syncExecution(video.currentTime); drawViews(); paint(); });
  video.addEventListener('play', paintControls);
  video.addEventListener('pause', () => { if (!video.ended) execution.pause(); paintControls(); });
  video.addEventListener('ended', () => { seek(0); playBoth(); });
  video.addEventListener('error', () => { pauseBoth(); setStatus('This example could not load. Select it again to retry.', true); });
  execution.addEventListener('loadedmetadata', () => { syncExecution(requestedTime, true); paintControls(); maybeAutoPlay(); });
  execution.addEventListener('loadeddata', () => { paintControls(); maybeAutoPlay(); });
  execution.addEventListener('error', () => { setStatus('The real-world video could not load.', true); paintControls(); });
  for (const media of [video, execution]) {
    media.addEventListener('waiting', () => {
      if (video.paused && execution.paused) return;
      pauseBoth();
      pendingAutoPlay = true;
      setStatus(window.GALATEA_MEDIA.message([video, execution]));
      paintControls();
    });
    for (const event of ['canplay', 'progress', 'seeked']) media.addEventListener(event, maybeAutoPlay);
  }
  video.addEventListener('seeked', () => {
    displayedTime = video.currentTime;
    drawViews();
    seekReference();
    paint();
    maybeAutoPlay();
  });
  playButton.onclick = () => pendingAutoPlay || !video.paused ? pauseBoth() : playBoth();
  scrub.oninput = (event) => {
    pauseBoth();
    seek(Number(event.target.value));
  };
  scrub.onkeydown = (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    pauseBoth();
    const direction = ['ArrowRight', 'ArrowUp'].includes(event.key) ? 1 : -1;
    seek(event.key === 'Home' ? 0 : event.key === 'End' ? duration() : (Math.floor(video.currentTime * selected.fps + 1e-4) + direction) / selected.fps);
  };
  root.addEventListener('keydown', (event) => {
    if (dialog.open || ['INPUT', 'SELECT', 'BUTTON', 'A', 'SUMMARY', 'VIDEO'].includes(event.target.tagName)) return;
    if (event.code === 'Space') { event.preventDefault(); video.paused ? playBoth() : pauseBoth(); }
  });
  byId('close').onclick = () => dialog.close();
  dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => {
    modalVideo.pause();
    modalVideo.onloadedmetadata = null;
    if (modalFrameCallback !== null && modalVideo.cancelVideoFrameCallback) modalVideo.cancelVideoFrameCallback(modalFrameCallback);
    modalFrameCallback = null;
    modalVideo.removeAttribute('src');
    modalVideo.load();
    pendingAutoPlay = true;
    maybeAutoPlay();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pauseBoth();
    else { pendingAutoPlay = true; maybeAutoPlay(); }
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      nearView = entries[0].isIntersecting;
      if (nearView) loadSelectedMedia();
    }, { rootMargin: '300px 0px' }).observe(root);
    new IntersectionObserver((entries) => {
      inView = entries[0].isIntersecting;
      if (inView) { pendingAutoPlay = true; maybeAutoPlay(); }
      else pauseBoth();
    }).observe(root);
  } else {
    inView = true;
  }
  select(data.samples[0]);
})();
