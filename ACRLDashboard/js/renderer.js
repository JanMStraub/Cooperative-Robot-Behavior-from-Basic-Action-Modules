export class Renderer {
    constructor() {
        this.meshCache = {};
        this.meshTimestamps = {};
        this.initThreeJS();
    }

    initThreeJS() {
        const container = document.getElementById('threejs-container');
        if (!container) return;
        
        const width = container.clientWidth;
        const height = container.clientHeight;

        const isLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(isLight ? 0xf0f2f5 : 0x0a0a0a);

        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        this.camera.position.set(0, 2, 3);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        const gridHelper = new THREE.GridHelper(5, 20, 0x4361ee, isLight ? 0xcccccc : 0x222222);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        const groundGeo = new THREE.PlaneGeometry(10, 10);
        const groundMat = new THREE.MeshStandardMaterial({ 
            color: isLight ? 0xf0f2f5 : 0x111216, 
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 8, 5);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 1024;
        dirLight.shadow.mapSize.height = 1024;
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 15;
        this.scene.add(dirLight);

        const fillLight = new THREE.DirectionalLight(0x90b0d0, 0.3);
        fillLight.position.set(-5, 3, -5);
        this.scene.add(fillLight);

        window.addEventListener('resize', () => {
            const newWidth = container.clientWidth;
            const newHeight = container.clientHeight;
            this.renderer.setSize(newWidth, newHeight);
            this.camera.aspect = newWidth / newHeight;
            this.camera.updateProjectionMatrix();
        });

        const animate = () => {
            requestAnimationFrame(animate);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }

    updateWorldState(data) {
        // ── DOM world state panel ──────────────────────────────────
        const robotList = document.getElementById('ws-robot-list');
        const objectSection = document.getElementById('ws-object-section');
        const objectList = document.getElementById('ws-object-list');
        const tsEl = document.getElementById('ws-timestamp');

        // Timestamp
        if (tsEl) {
            const now = new Date();
            tsEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
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
                const moving  = robot.is_moving;
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
                            <span style="font-size:0.72rem;color:var(--text-muted)">${moving ? 'Moving…' : 'Idle'}</span>
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
                objectSection.style.display = '';
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

    updateOrCreateMesh(id, data, type) {
        let mesh = this.meshCache[id];

        if (!mesh) {
            if (type === 'robot') {
                mesh = new THREE.Group();

                // ---- Base disc ----
                const baseGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.04, 32);
                const baseMat = new THREE.MeshStandardMaterial({ color: 0x222244, roughness: 0.5, metalness: 0.6 });
                const base = new THREE.Mesh(baseGeo, baseMat);
                base.position.y = 0.02;
                base.castShadow = true;
                mesh.add(base);

                // ---- Arm column ----
                const armGeo = new THREE.CylinderGeometry(0.04, 0.07, 0.28, 16);
                const armMat = new THREE.MeshStandardMaterial({ color: 0xff6a00, roughness: 0.4, metalness: 0.5 });
                const arm = new THREE.Mesh(armGeo, armMat);
                arm.position.y = 0.18;
                arm.castShadow = true;
                mesh.add(arm);

                // ---- Shoulder joint ----
                const shoulderGeo = new THREE.SphereGeometry(0.055, 16, 16);
                const jointMat = new THREE.MeshStandardMaterial({ color: 0x333355, roughness: 0.5, metalness: 0.7 });
                const shoulder = new THREE.Mesh(shoulderGeo, jointMat);
                shoulder.position.y = 0.33;
                shoulder.castShadow = true;
                mesh.add(shoulder);

                // ---- Forearm ----
                const forearmGeo = new THREE.CylinderGeometry(0.03, 0.04, 0.22, 16);
                const forearm = new THREE.Mesh(forearmGeo, armMat.clone());
                forearm.position.y = 0.46;
                forearm.castShadow = true;
                mesh.add(forearm);

                // ---- End-effector ----
                const eeGeo = new THREE.SphereGeometry(0.04, 16, 16);
                const eeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.8 });
                const ee = new THREE.Mesh(eeGeo, eeMat);
                ee.position.y = 0.59;
                ee.castShadow = true;
                mesh.add(ee);
                // Store reference for gripper state color change
                mesh.userData.endEffector = ee;
                mesh.userData.eeMat = eeMat;
            } else {
                let colorHex = 0xaaaaaa; // Default gray for unknown objects
                let isTransparent = false;
                let opacity = 1.0;

                if (data.color) {
                    const col = data.color.toLowerCase();
                    if (col.includes('blue')) colorHex = 0x4361ee;
                    else if (col.includes('green')) colorHex = 0x2ec4b6;
                    else if (col.includes('yellow')) colorHex = 0xff9f1c;
                    else if (col.includes('orange')) colorHex = 0xf58231;
                    else if (col.includes('red')) colorHex = 0xe71d36;
                    else if (col.includes('purple')) colorHex = 0x9b5de5;
                    else if (col.includes('white')) colorHex = 0xffffff;
                    else if (col.includes('black')) colorHex = 0x222222;
                    else if (col.includes('field') || col.includes('table')) {
                        colorHex = 0xcccccc;
                        isTransparent = true;
                        opacity = 0.3;
                    }
                }

                if (data.object_id) {
                    const id = data.object_id.toLowerCase();
                    if (id.includes('table') || id.includes('workspace') || id.includes('floor') || id.includes('plane') || id.includes('base')) {
                        isTransparent = true;
                        opacity = 0.3;
                    }
                }

                const geo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
                const mat = new THREE.MeshStandardMaterial({ 
                    color: colorHex,
                    roughness: 0.7,
                    metalness: 0.1,
                    transparent: isTransparent,
                    opacity: opacity
                });
                mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = !isTransparent;
                mesh.receiveShadow = true;
            }

            this.scene.add(mesh);
            this.meshCache[id] = mesh;
        }

        if (data.position) {
            if (Array.isArray(data.position) && data.position.length >= 3) {
                mesh.position.set(data.position[0], data.position[1], data.position[2]);
            } else if (data.position.x !== undefined && data.position.y !== undefined && data.position.z !== undefined) {
                mesh.position.set(data.position.x, data.position.y, data.position.z);
            }
        }

        if (data.rotation) {
            if (Array.isArray(data.rotation) && data.rotation.length >= 3) {
                mesh.rotation.set(
                    THREE.MathUtils.degToRad(data.rotation[0]),
                    THREE.MathUtils.degToRad(data.rotation[1]),
                    THREE.MathUtils.degToRad(data.rotation[2])
                );
            } else if (data.rotation.x !== undefined && data.rotation.w !== undefined) {
                mesh.quaternion.set(data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w);
            }
        }

        if (type === 'object' && data.dimensions) {
            if (Array.isArray(data.dimensions) && data.dimensions.length >= 3) {
                mesh.scale.set(data.dimensions[0] / 0.1, data.dimensions[1] / 0.1, data.dimensions[2] / 0.1);
            } else if (data.dimensions.x !== undefined) {
                mesh.scale.set(data.dimensions.x / 0.1, data.dimensions.y / 0.1, data.dimensions.z / 0.1);
            }
        }

        if (type === 'robot' && data.gripper_state !== undefined && mesh.userData.eeMat) {
            // Visualize gripper state via end-effector color: green=open, red=closed
            mesh.userData.eeMat.color.set(data.gripper_state === 'open' ? 0x2ec4b6 : 0xe71d36);
        }
    }
}
