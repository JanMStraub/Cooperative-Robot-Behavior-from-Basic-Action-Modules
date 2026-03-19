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
        if (!this.scene) return;
        const now = Date.now();

        if (data.objects) {
            data.objects.forEach(obj => {
                const id = obj.object_id;
                this.meshTimestamps[id] = now;
                this.updateOrCreateMesh(id, obj, 'object');
            });
        }

        if (data.robots) {
            data.robots.forEach(robot => {
                const id = robot.robot_id;
                this.meshTimestamps[id] = now;
                this.updateOrCreateMesh(id, robot, 'robot');
            });
        }

        Object.keys(this.meshCache).forEach(id => {
            const lastSeen = this.meshTimestamps[id] || 0;
            if (now - lastSeen > 2000) {
                this.scene.remove(this.meshCache[id]);
                delete this.meshCache[id];
                delete this.meshTimestamps[id];
            }
        });
    }

    updateOrCreateMesh(id, data, type) {
        let mesh = this.meshCache[id];

        if (!mesh) {
            if (type === 'robot') {
                mesh = new THREE.Group();

                if (!this.urdfLoader && window.URDFLoader) {
                    this.urdfLoader = new window.URDFLoader(new THREE.LoadingManager());
                    // Update package URLs to point to our mounted /urdf route
                    // and correctly handle case sensitivity for .STL files
                    this.urdfLoader.parsePackageUrl = (url) => {
                        let replaced = url.replace('package://ar4_stl/', '/urdf/ar4_stl/');
                        if (/(base_link|link_[1-6])\.stl$/.test(replaced)) {
                           replaced = replaced.replace(/\.stl$/, '.STL');
                        }
                        return replaced;
                    };
                }

                if (this.urdfLoader) {
                    this.urdfLoader.load('/urdf/ar4.urdf', (robot) => {
                        robot.rotation.x = -Math.PI / 2; // Convert Z-up to Y-up
                        
                        // Scale the robot appropriately for the world view
                        robot.scale.set(1, 1, 1);
                        
                        // Apply shadows and simple materials since STL lacks them
                        const bodyMaterial = new THREE.MeshStandardMaterial({ 
                            color: 0xdddddd, 
                            roughness: 0.5, 
                            metalness: 0.2 
                        });
                        const jointMaterial = new THREE.MeshStandardMaterial({ 
                            color: 0x333333, 
                            roughness: 0.7, 
                            metalness: 0.1 
                        });

                        robot.traverse(child => {
                            if (child.isMesh) {
                                child.castShadow = true;
                                child.receiveShadow = true;
                                // Basic heuristic: if the link name contains 'link' it's body, otherwise joint/gripper
                                if (child.parent && child.parent.name.includes('link')) {
                                    child.material = bodyMaterial.clone();
                                } else {
                                    child.material = jointMaterial.clone();
                                }
                            }
                        });
                        mesh.add(robot);
                    });
                }
            } else {
                let colorHex = 0xe71d36; 
                if (data.color) {
                    if (data.color.toLowerCase().includes('blue')) colorHex = 0x4361ee;
                    else if (data.color.toLowerCase().includes('green')) colorHex = 0x2ec4b6;
                    else if (data.color.toLowerCase().includes('yellow')) colorHex = 0xff9f1c;
                    else if (data.color.toLowerCase().includes('orange')) colorHex = 0xf58231;
                    else if (data.color.toLowerCase().includes('field')) colorHex = 0x888888;
                }
                const geo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
                const mat = new THREE.MeshStandardMaterial({ 
                    color: colorHex,
                    roughness: 0.7,
                    metalness: 0.1
                });
                mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = true;
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

        if (type === 'robot' && data.joint_angles && mesh.children.length > 0) {
            const urdfRobot = mesh.children[0];
            if (urdfRobot.isURDFRobot) {
                // The URDF joints are named joint_1 to joint_6 and gripper_jaw1_joint/gripper_jaw2_joint
                for (let i = 0; i < data.joint_angles.length; i++) {
                    const jointName = `joint_${i + 1}`;
                    if (urdfRobot.joints[jointName]) {
                        urdfRobot.setJointValue(jointName, data.joint_angles[i]);
                    }
                }
                // Handle Gripper
                if (data.gripper_state !== undefined) {
                    const gripperPos = data.gripper_state === 'open' ? 0.014 : 0;
                    if (urdfRobot.joints['gripper_jaw1_joint']) urdfRobot.setJointValue('gripper_jaw1_joint', gripperPos);
                    if (urdfRobot.joints['gripper_jaw2_joint']) urdfRobot.setJointValue('gripper_jaw2_joint', gripperPos);
                }
            }
        }
    }
}
