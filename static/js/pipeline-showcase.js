/* Synchronized Highlights playback: one reference clock drives the reconstruction
   views and a keyframe-mapped, accelerated real-world recording. */
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
  const stages = [
    { id: 'input', step: 'GENERATE', label: 'Generated video', note: 'The generated RGB video used by the reconstruction pipeline.' },
    { id: 'depth', step: 'GEOMETRY', label: 'MoGe-2', note: 'Background-calibrated monocular depth, with a fixed color scale for this sequence.' },
    { id: 'sam2', step: 'SEGMENTATION', label: 'SAM2', note: 'The selected target-object mask overlaid on the input video.' },
    { id: 'trajectory', step: 'MOTION', label: 'Hand–object motion', note: 'The reconstructed hand and object in a shared camera view.' }
  ];
  const executionStage = { id: 'execution', label: 'Real-world execution', note: 'The physical robot execution synchronized to the generated reference.' };
  const phaseLabels = { pre: 'Pre-contact', contact: 'Contact', post: 'Post-contact', gap: 'No contact' };
  const views = [];
  let selected = data.samples[0];
  let generation = 0;
  let playRequest = 0;
  let animation = null;
  let displayedTime = null;
  let requestedTime = 0;
  let pendingSeek = false;
  let speed = 1;
  let inView = false;
  let nearView = !('IntersectionObserver' in window);
  let pendingAutoPlay = false;
  let modalFrameCallback = null;

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
        return Math.max(.1, (points[i][1] - points[i - 1][1]) / Math.max(.0001, points[i][0] - points[i - 1][0]) * speed);
      }
    }
    return Math.max(.1, speed);
  }

  function phaseAt(sample, time) {
    const frame = Math.max(0, Math.min(sample.frames - 1, Math.floor(time * sample.fps + 1e-4)));
    if (sample.contact.some(([a, b]) => frame >= a && frame <= b)) return 'contact';
    return frame < sample.contact[0][0] ? 'pre' : frame > sample.contact.at(-1)[1] ? 'post' : 'gap';
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
    const phase = phaseAt(selected, displayedTime ?? time);
    byId('panel').dataset.phase = phase;
    byId('phase').textContent = phaseLabels[phase];
    byId('scrub').value = time;
    byId('scrub').setAttribute('aria-valuetext', `${time.toFixed(2)} of ${duration().toFixed(2)} seconds, ${phaseLabels[phase]}`);
    byId('time').textContent = `${time.toFixed(2)} / ${duration().toFixed(2)} s`;
    byId('cursor').style.left = `${time / duration() * 100}%`;
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
    playButton.disabled = !ready;
    byId('restart-both').disabled = !(video.readyState >= 1 && execution.readyState >= 1);
    playButton.textContent = pendingAutoPlay && video.paused ? 'Cancel autoplay' : video.paused ? 'Play synchronized' : 'Pause synchronized';
    playButton.setAttribute('aria-label', pendingAutoPlay && video.paused ? 'Cancel autoplay' : video.paused ? 'Play generated and real-world videos in sync' : 'Pause synchronized videos');
  }

  function pauseBoth() {
    if (pendingAutoPlay && !pendingSeek) byId('status').textContent = 'Paused';
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
    if (pendingSeek || !window.GALATEA_MEDIA.ready([video, execution])) {
      byId('status').textContent = window.GALATEA_MEDIA.message([video, execution]);
      paintControls();
      return;
    }
    pendingAutoPlay = false;
    byId('status').textContent = '';
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
      byId('status').textContent = 'Playback was interrupted. Select Play synchronized to try again.';
    }
  }

  function seek(time) {
    requestedTime = Math.max(0, Math.min(time, duration()));
    pendingSeek = true;
    byId('status').textContent = 'Loading selected frame…';
    seekReference();
    paint();
  }

  function seekReference() {
    if (!pendingSeek || video.readyState < 1 || video.seeking) return;
    const target = frameTarget(selected, requestedTime);
    if (Math.abs(video.currentTime - target) < .00001) {
      pendingSeek = false;
      byId('status').textContent = '';
      displayedTime = video.currentTime;
      drawViews();
      paint();
      return;
    }
    // Let decoding finish before applying the latest drag position.
    video.currentTime = target;
  }

  function maybeAutoPlay() {
    if (pendingSeek) byId('status').textContent = 'Loading selected frame…';
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
    byId('segments').innerHTML = spans.map(([a, b, phase]) => `<span class="ps-segment" data-phase="${phase}" style="left:${a / duration() * 100}%;width:${(b - a) / duration() * 100}%"></span>`).join('');
    byId('segments').title = `Contact frames (inclusive): ${selected.contact.map(([a, b]) => `${a}–${b}`).join(', ')}`;
    byId('scrub').max = duration();
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
    displayedTime = null;
    requestedTime = 0;
    byId('panel').dataset.sample = selected.id;
    byId('samples').querySelectorAll('button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.sample === selected.id)));
    byId('action').textContent = selected.execution.action;
    byId('status').textContent = 'Loading reconstruction…';
    byId('execution-status').textContent = 'Keyframe synchronization loading…';
    byId('scrub').disabled = true;
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
    video.playbackRate = speed;
    execution.playbackRate = executionRate(0);
    video.preload = execution.preload = 'none';
    video.load();
    execution.load();
    if (nearView) loadSelectedMedia();
    video.setAttribute('aria-label', `${selected.label}: generated video synchronized with reconstruction and real-world execution`);
    pendingAutoPlay = true;
    paint();
    paintControls();
  }

  function enlarge(stage) {
    const time = video.currentTime;
    pauseBoth();
    byId('dialog-title').textContent = `${selected.label} / ${stage.label}`;
    byId('dialog-note').textContent = stage.note;
    const source = stage.id === 'execution' ? selected.execution : selected.media[stage.id];
    modalVideo.poster = url(source.poster);
    modalVideo.onloadedmetadata = () => {
      if (!dialog.open) return;
      modalVideo.currentTime = stage.id === 'execution' ? mapExecutionTime(time) : frameTarget(selected, time);
      modalVideo.playbackRate = stage.id === 'execution' ? executionRate(time) : speed;
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

  byId('samples').innerHTML = data.samples.map((sample) => `<button type="button" class="ps-sample" data-sample="${sample.id}" aria-pressed="false" aria-label="Select ${sample.label}">${sample.label}</button>`).join('');
  byId('samples').querySelectorAll('button').forEach((button) => button.onclick = () => select(data.samples.find((sample) => sample.id === button.dataset.sample)));
  byId('stages').innerHTML = stages.slice(1).map((stage) => `<button type="button" class="ps-stage-button" data-stage="${stage.id}" aria-label="Enlarge ${stage.label}"><span class="ps-stage-screen"><img alt="" aria-hidden="true"><canvas width="424" height="240" aria-hidden="true"></canvas><span class="ps-expand" aria-hidden="true">↗</span></span><span class="ps-stage-caption">${stage.label}</span></button>`).join('');
  byId('stages').querySelectorAll('button').forEach((button) => {
    const canvas = button.querySelector('canvas');
    views.push({ stage: button.dataset.stage, canvas, context: canvas.getContext('2d'), poster: button.querySelector('img') });
  });
  root.querySelectorAll('[data-stage]').forEach((button) => button.onclick = () => enlarge(stages.find((stage) => stage.id === button.dataset.stage) || executionStage));

  video.addEventListener('loadedmetadata', () => { video.currentTime = frameTarget(selected, requestedTime); video.playbackRate = speed; syncExecution(requestedTime, true); paint(); });
  video.addEventListener('loadeddata', () => { byId('status').textContent = ''; byId('scrub').disabled = false; drawViews(); paint(); paintControls(); maybeAutoPlay(); });
  video.addEventListener('timeupdate', () => { if (pendingSeek || video.seeking) return; displayedTime = video.currentTime; syncExecution(video.currentTime); drawViews(); paint(); });
  video.addEventListener('play', paintControls);
  video.addEventListener('pause', () => { if (!video.ended) execution.pause(); paintControls(); });
  video.addEventListener('ended', () => { if (byId('loop').checked) { seek(0); playBoth(); } else pauseBoth(); });
  video.addEventListener('error', () => { pauseBoth(); byId('status').textContent = 'Reconstruction could not load. Select the example again to retry.'; });
  execution.addEventListener('loadedmetadata', () => { syncExecution(requestedTime, true); byId('execution-status').textContent = ''; paintControls(); maybeAutoPlay(); });
  execution.addEventListener('loadeddata', () => { byId('execution-status').textContent = ''; paintControls(); maybeAutoPlay(); });
  execution.addEventListener('error', () => { byId('execution-status').textContent = 'Execution could not load.'; paintControls(); });
  for (const media of [video, execution]) {
    media.addEventListener('waiting', () => {
      if (video.paused && execution.paused) return;
      pauseBoth();
      pendingAutoPlay = true;
      byId('status').textContent = window.GALATEA_MEDIA.message([video, execution]);
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
  byId('restart-both').onclick = () => { pauseBoth(); seek(0); };
  byId('scrub').oninput = (event) => {
    const time = Number(event.target.value);
    pauseBoth();
    seek(time);
  };
  byId('speed').onchange = (event) => { speed = Number(event.target.value); video.playbackRate = speed; syncExecution(video.currentTime, true); };
  byId('scrub').onkeydown = (event) => {
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
  dialog.addEventListener('close', () => {
    modalVideo.pause();
    modalVideo.onloadedmetadata = null;
    if (modalFrameCallback !== null && modalVideo.cancelVideoFrameCallback) modalVideo.cancelVideoFrameCallback(modalFrameCallback);
    modalFrameCallback = null;
    modalVideo.removeAttribute('src');
    modalVideo.load();
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
