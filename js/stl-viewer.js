// STL Viewer functionality using Three.js
function createSTLViewer(container, stlPath) {
    // Clear any existing content
    container.innerHTML = '';
    
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5);
    
    // Camera setup - positioned further back initially
    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 10); // Position camera further back
    
    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
    scene.add(ambientLight);
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight1.position.set(1, 1, 1).normalize();
    scene.add(directionalLight1);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight2.position.set(-1, -1, -1).normalize();
    scene.add(directionalLight2);
    
    // Controls for rotating the model
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.enableZoom = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 2.0;
    
    // Show loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'stl-loading';
    loadingDiv.innerHTML = '<p>Loading model...</p>';
    container.appendChild(loadingDiv);
    
    // Load STL file
    const loader = new THREE.STLLoader();
    loader.load(
        stlPath, 
        function (geometry) {
            // Remove loading indicator
            const loadingElement = container.querySelector('.stl-loading');
            if (loadingElement) {
                loadingElement.remove();
            }
            
            // Create material based on file name
            let color = 0x3498db; // Default blue
            
            // Assign different colors based on model type
            if (stlPath.includes('EMG')) {
                color = 0x2ecc71; // Green for EMG
            } else if (stlPath.includes('IMUBase')) {
                color = 0xe74c3c; // Red for IMU Base
            } else if (stlPath.includes('IMUTop')) {
                color = 0xf39c12; // Orange for IMU Top
            }
            
            const material = new THREE.MeshPhongMaterial({ 
                color: color, 
                specular: 0x111111, 
                shininess: 200 
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            
            // Center the model - improved centering
            geometry.computeBoundingBox();
            const box = geometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);
            
            // Instead of changing mesh position, we move all geometry vertices
            geometry.translate(-center.x, -center.y, -center.z);
            
            // Scale the model to fit the viewer
            const maxDim = Math.max(
                box.max.x - box.min.x,
                box.max.y - box.min.y,
                box.max.z - box.min.z
            );
            const scale = 3.5 / maxDim; // Slightly smaller scale to ensure visibility
            mesh.scale.set(scale, scale, scale);
            
            // Add mesh to scene
            scene.add(mesh);
            
            // Set up camera to look at center of model
            camera.lookAt(scene.position);
            
            // Reset camera and controls to better view the model
            controls.reset();
            
            // Add animation loop
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        },
        // Progress callback
        function (xhr) {
            const loadingElement = container.querySelector('.stl-loading');
            if (loadingElement) {
                loadingElement.innerHTML = `<p>Loading: ${Math.round(xhr.loaded / xhr.total * 100)}%</p>`;
            }
        },
        // Error callback
        function (error) {
            console.error(`Error loading ${stlPath}:`, error);
            // Show error in viewer
            const loadingElement = container.querySelector('.stl-loading');
            if (loadingElement) {
                loadingElement.innerHTML = `<p>Error loading model</p>`;
                loadingElement.className = 'stl-error';
            }
        }
    );
    
    // Handle window resize
    window.addEventListener('resize', function() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// Initialize STL viewers when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .stl-loading, .stl-error {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10;
            pointer-events: none;
        }
        
        .stl-loading {
            background-color: rgba(255, 255, 255, 0.7);
            color: #3498db;
        }
        
        .stl-error {
            background-color: rgba(255, 236, 236, 0.7);
            color: #e74c3c;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize viewers
    if (typeof THREE === 'undefined') {
        console.warn('THREE is not defined, waiting...');
        setTimeout(initSTLViewers, 500);
    } else {
        initSTLViewers();
    }
});

function initSTLViewers() {
    const stlViewers = document.querySelectorAll('.stl-viewer');
    
    stlViewers.forEach((viewer) => {
        const stlPath = viewer.getAttribute('data-stl-path');
        if (stlPath) {
            createSTLViewer(viewer, stlPath);
        }
    });
}