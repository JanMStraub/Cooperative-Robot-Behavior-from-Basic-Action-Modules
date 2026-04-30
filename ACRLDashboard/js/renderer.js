export class Renderer {
    constructor() {
        this.meshCache = {};
        this.meshTimestamps = {};
    }

    updateWorldState(data) {
        if (this.wsConnectionTimeout) {
            clearTimeout(this.wsConnectionTimeout);
            this.wsConnectionTimeout = null;
        }
        // ── DOM world state panel ──────────────────────────────────
        const robotList = document.getElementById('ws-robot-list');
        const objectSection = document.getElementById('ws-object-section');
        const objectList = document.getElementById('ws-object-list');
        const tsEl = document.getElementById('ws-timestamp');

        // Timestamp
        if (tsEl) {
            const now = new Date();
            tsEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        }

        // Robot cards
        if (robotList && data.robots && data.robots.length > 0) {
            robotList.innerHTML = '';
            data.robots.forEach(robot => {
                const pos  = robot.position || {};
                const tgt  = robot.target_position || {};
                const f = v => (v != null ? Number(v).toFixed(3) : '–');
                const px = f(Array.isArray(pos) ? pos[0] : pos.x);
                const py = f(Array.isArray(pos) ? pos[1] : pos.y);
                const pz = f(Array.isArray(pos) ? pos[2] : pos.z);
                const tx = f(Array.isArray(tgt) ? tgt[0] : tgt.x);
                const ty = f(Array.isArray(tgt) ? tgt[1] : tgt.y);
                const tz = f(Array.isArray(tgt) ? tgt[2] : tgt.z);
                const moving  = robot.is_moving === true;
                const gripper = (robot.gripper_state || 'unknown').toLowerCase();
                const mode    = robot.control_mode;
                const joints  = robot.joint_angles;

                const modeTag  = mode  ? `<span class="ws-mode-badge">${mode}</span>` : '';
                const jointStr = joints && joints.length
                    ? joints.map((j, i) => `<span><span class="ws-pos-label">J${i+1}</span> ${Number(j).toFixed(2)}</span>`).join('')
                    : '';

                const card = document.createElement('div');
                card.className = `ws-robot-card${moving ? ' moving' : ''}`;
                card.innerHTML = `
                    <div class="ws-robot-header">
                        <span class="ws-robot-name">
                            <i class="fa-solid fa-robot"></i>
                            ${robot.robot_id}
                            ${modeTag}
                        </span>
                        <div style="display:flex;align-items:center;gap:0.5rem;">
                            <span style="font-size:0.72rem;color:${moving ? 'var(--warning)' : 'var(--text-muted)'}">${moving ? 'Moving…' : 'Idle'}</span>
                            <span class="ws-status-dot${moving ? ' moving' : ''}"></span>
                        </div>
                    </div>
                    <div class="ws-pos-row">
                        <span style="color:var(--text-muted);font-size:0.7rem;min-width:2rem">EE</span>
                        <span><span class="ws-pos-label">X</span> ${px}</span>
                        <span><span class="ws-pos-label">Y</span> ${py}</span>
                        <span><span class="ws-pos-label">Z</span> ${pz}</span>
                    </div>
                    <div class="ws-pos-row">
                        <span style="color:var(--text-muted);font-size:0.7rem;min-width:2rem">Tgt</span>
                        <span><span class="ws-pos-label">X</span> ${tx}</span>
                        <span><span class="ws-pos-label">Y</span> ${ty}</span>
                        <span><span class="ws-pos-label">Z</span> ${tz}</span>
                    </div>
                    <div class="ws-gripper-row">
                        <span style="color:var(--text-muted);font-size:0.78rem">Gripper:</span>
                        <span class="ws-gripper-badge ${gripper}">${gripper}</span>
                    </div>
                    ${jointStr ? `<div class="ws-pos-row ws-joint-row">${jointStr}</div>` : ''}`;
                robotList.appendChild(card);
            });
        } else if (robotList && (!data.robots || data.robots.length === 0)) {
            robotList.innerHTML = `<div class="ws-empty"><i class="fa-solid fa-circle-notch fa-spin"></i><span>Waiting for Unity…</span></div>`;
        }

        // Object cards
        if (objectSection && objectList && data.objects) {
            if (data.objects.length > 0) {
                objectSection.style.display = 'flex';
                objectList.innerHTML = '';
                data.objects.forEach(obj => {
                    const p = obj.position || {};
                    const d = obj.dimensions || {};
                    const f = v => (v != null ? Number(v).toFixed(3) : '–');
                    const ox = f(Array.isArray(p) ? p[0] : p.x);
                    const oy = f(Array.isArray(p) ? p[1] : p.y);
                    const oz = f(Array.isArray(p) ? p[2] : p.z);
                    const dw = f(Array.isArray(d) ? d[0] : d.x);
                    const dh = f(Array.isArray(d) ? d[1] : d.y);
                    const dd = f(Array.isArray(d) ? d[2] : d.z);
                    const color = (obj.color || 'unknown').toLowerCase();
                    const grasped = obj.grasped_by;
                    const colorDot = color !== 'unknown'
                        ? `<span class="ws-color-dot ws-color-${color}"></span>`
                        : '';
                    const graspedTag = grasped
                        ? `<span class="ws-grasped-badge"><i class="fa-solid fa-hand"></i> ${grasped}</span>`
                        : '';

                    const card = document.createElement('div');
                    card.className = 'ws-object-card';
                    card.innerHTML = `
                        <div class="ws-object-header">
                            ${colorDot}
                            <span class="ws-object-name">${obj.object_id || 'Object'}</span>
                            ${graspedTag}
                        </div>
                        <div class="ws-pos-row">
                            <span><span class="ws-pos-label">X</span> ${ox}</span>
                            <span><span class="ws-pos-label">Y</span> ${oy}</span>
                            <span><span class="ws-pos-label">Z</span> ${oz}</span>
                        </div>
                        <div class="ws-pos-row" style="opacity:0.6">
                            <span style="color:var(--text-muted);font-size:0.7rem;min-width:2rem">dim</span>
                            <span>${dw} × ${dh} × ${dd}</span>
                        </div>`;
                    objectList.appendChild(card);
                });
            } else {
                objectSection.style.display = 'none';
            }
        }
    }

    /* ── VGN Debug Visualization ── */
    createMiniRenderer(containerId, scene, camera) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Position camera to look at the table
        camera.position.set(0, 0.4, 0.6);
        controls.target.set(0, 0, 0);

        const applySize = () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            if (w === 0 || h === 0) return;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        };
        const resizeOb = new ResizeObserver(applySize);
        resizeOb.observe(container);
        // Flush layout before reading clientWidth/clientHeight (container just became visible)
        requestAnimationFrame(applySize);

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        return { renderer, controls };
    }

    updateStereoPointCloud(data) {
        const container = document.getElementById('stereo-pc-container');
        if (!container) return;
        container.style.display = 'block';

        if (!this.stereoPCInited) {
            this.stereoPCInited = true;
            const placeholder = document.getElementById('stereo-pc-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            this.stereoPCScene = new THREE.Scene();
            this.stereoPCScene.add(new THREE.GridHelper(2.0, 40, 0x444444, 0x222222));
            this.stereoPCScene.add(new THREE.AmbientLight(0xffffff, 1.0));
            this.stereoPCCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
            this.createMiniRenderer('stereo-pc-container', this.stereoPCScene, this.stereoPCCamera);
        }

        if (!data.points_b64) return;

        // Decode XYZ (float32) and RGB (uint8) binary blobs
        const ptsBin = atob(data.points_b64);
        const ptsBytes = new Uint8Array(ptsBin.length);
        for (let i = 0; i < ptsBin.length; i++) ptsBytes[i] = ptsBin.charCodeAt(i);
        const positions = new Float32Array(ptsBytes.buffer);
        const n = positions.length / 3;

        let colors = null;
        if (data.colors_b64) {
            const clrBin = atob(data.colors_b64);
            const clrBytes = new Uint8Array(clrBin.length);
            for (let i = 0; i < clrBin.length; i++) clrBytes[i] = clrBin.charCodeAt(i);
            // Normalise uint8 [0,255] → float [0,1] for Three.js vertex colors
            colors = new Float32Array(clrBytes.length);
            for (let i = 0; i < clrBytes.length; i++) colors[i] = clrBytes[i] / 255.0;
        }

        // Points arrive in Unity LH world frame (+X right, +Y up, +Z forward).
        // Three.js is RH +Y up: negate Z to convert.
        for (let i = 2; i < positions.length; i += 3) positions[i] = -positions[i];

        // Compute centroid for camera framing
        let cx = 0, cy = 0, cz = 0;
        for (let i = 0; i < positions.length; i += 3) { cx += positions[i]; cy += positions[i+1]; cz += positions[i+2]; }
        cx /= n; cy /= n; cz /= n;
        const span = data.scene_span || 1.5;
        this.stereoPCCamera.position.set(cx, cy + span * 0.6, cz + span * 1.2);
        this.stereoPCCamera.lookAt(cx, cy, cz);

        if (this.stereoPCMesh) this.stereoPCScene.remove(this.stereoPCMesh);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        if (colors) geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const mat = new THREE.PointsMaterial({
            size: Math.max(span * 0.002, 0.003),
            vertexColors: !!colors,
            color: colors ? 0xffffff : 0x2ec4b6,
        });
        this.stereoPCMesh = new THREE.Points(geo, mat);
        this.stereoPCScene.add(this.stereoPCMesh);
    }
}
