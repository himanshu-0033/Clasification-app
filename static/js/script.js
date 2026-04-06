// Compliments Array
const compliments = [
    "You look amazing today!",
    "You're doing better than you think.",
    "You're a whole vibe 😌",
    "Your smile lights up the room ✨",
    "You are enough, exactly as you are.",
    "Take a deep breath. You got this.",
    "You make the world a little brighter 💖",
    "Someone is very lucky to have you in their life.",
    "I'm so proud of how far you've come.",
    "You are beautifully unique."
];

const smileBtn = document.getElementById('smile-btn');
const complimentText = document.getElementById('compliment-text');

smileBtn.addEventListener('click', () => {
    // Generate random compliment
    const randomIndex = Math.floor(Math.random() * compliments.length);
    const text = compliments[randomIndex];
    
    // Animate
    complimentText.classList.remove('show');
    
    setTimeout(() => {
        complimentText.textContent = text;
        complimentText.classList.add('show');
    }, 300); // Wait for fade out
});

// Pookie Mirror Canvas
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const mirrorInput = document.getElementById('mirror-input');
const clearBtn = document.getElementById('clear-btn');

function resizeCanvas() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas(); // Initial setup

let isDrawing = false;

mirrorInput.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

mirrorInput.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    
    ctx.lineTo(e.offsetX, e.offsetY);
    
    // Get current theme color for drawing
    const computedStyle = getComputedStyle(document.documentElement);
    let drawColor = computedStyle.getPropertyValue('--primary').trim();
    if (!drawColor) {
        // Fallback or explicit check if variable is not read properly
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        drawColor = isDark ? '#00f0ff' : '#ffb7b2';
    }
    
    ctx.strokeStyle = drawColor;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Add glow effect for drawing
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (isDark) {
        ctx.shadowBlur = 10;
        ctx.shadowColor = drawColor;
    } else {
        ctx.shadowBlur = 0;
    }
    
    ctx.stroke();
});

window.addEventListener('mouseup', () => {
    isDrawing = false;
});

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    mirrorInput.value = '';
});

// Custom Cursor
const cursor = document.querySelector('.custom-cursor');
let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;
let cursorX = window.innerWidth / 2;
let cursorY = window.innerHeight / 2;

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    cursor.style.opacity = '1';
});

function animateCursor() {
    // Smooth follow
    cursorX += (mouseX - cursorX) * 0.2;
    cursorY += (mouseY - cursorY) * 0.2;
    
    cursor.style.transform = `translate(${cursorX - 10}px, ${cursorY - 10}px)`;
    requestAnimationFrame(animateCursor);
}
animateCursor();

// Add hover effects to cursor
const interactables = document.querySelectorAll('button, textarea');
interactables.forEach(el => {
    el.addEventListener('mouseenter', () => {
        cursor.style.width = '40px';
        cursor.style.height = '40px';
    });
    el.addEventListener('mouseleave', () => {
        cursor.style.width = '20px';
        cursor.style.height = '20px';
    });
});

// Background Particles Animation
const bgCanvas = document.getElementById('bg-canvas');
const bgCtx = bgCanvas.getContext('2d');

function initBgCanvas() {
    bgCanvas.width = window.innerWidth;
    bgCanvas.height = window.innerHeight;
}

window.addEventListener('resize', initBgCanvas);
initBgCanvas();

const particles = [];
const particleCount = 40;

class Particle {
    constructor() {
        this.reset();
        this.y = Math.random() * bgCanvas.height; 
    }
    
    reset() {
        this.x = Math.random() * bgCanvas.width;
        this.y = bgCanvas.height + 10;
        this.size = Math.random() * 3 + 1;
        this.speedY = Math.random() * -0.5 - 0.2;
        this.speedX = Math.random() * 0.4 - 0.2;
        this.opacity = Math.random() * 0.5 + 0.1;
    }
    
    update() {
        this.y += this.speedY;
        this.x += this.speedX;
        
        // Wrap around
        if (this.y < -10) {
            this.reset();
        }
    }
    
    draw() {
        bgCtx.beginPath();
        bgCtx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        bgCtx.fillStyle = isDark ? `rgba(0, 240, 255, ${this.opacity})` : `rgba(255, 183, 178, ${this.opacity})`;
        
        if (isDark) {
            bgCtx.shadowBlur = 10;
            bgCtx.shadowColor = '#00f0ff';
        } else {
            bgCtx.shadowBlur = 0;
        }
        
        bgCtx.fill();
    }
}

for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
}

function animateParticles() {
    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
    
    for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
    }
    
    requestAnimationFrame(animateParticles);
}

animateParticles();

// Webcam Mirror & MediaPipe Hand Tracking (Drawing in the air!)
const videoContainer = document.getElementById('mirror-video');

function initHandTracking() {
    try {
        const hands = new Hands({locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }});
        
        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
        });
        
        let lastHandX = null;
        let lastHandY = null;
        
        hands.onResults((results) => {
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                const landmarks = results.multiHandLandmarks[0];
                const indexTip = landmarks[8];
                const thumbTip = landmarks[4];
                
                // Pinch detection (distance between thumb tip and index finger tip)
                const isPinching = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y) < 0.08; 
                
                // Map the normalized coordinates to canvas coordinates
                // We do (1 - x) because the video is mirrored.
                const x = (1 - indexTip.x) * canvas.width;
                const y = indexTip.y * canvas.height;
                
                if (isPinching) {
                    if (lastHandX === null || lastHandY === null) {
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                        
                        const computedStyle = getComputedStyle(document.documentElement);
                        let drawColor = computedStyle.getPropertyValue('--primary').trim();
                        if (!drawColor) {
                            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                            drawColor = isDark ? '#00f0ff' : '#ffb7b2';
                        }
                        
                        ctx.strokeStyle = drawColor;
                        ctx.lineWidth = 5;
                        ctx.lineCap = 'round';
                        ctx.lineJoin = 'round';
                        
                        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                        if (isDark) {
                            ctx.shadowBlur = 15;
                            ctx.shadowColor = drawColor;
                        } else {
                            ctx.shadowBlur = 0;
                        }
                        
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                    }
                    lastHandX = x;
                    lastHandY = y;
                } else {
                    lastHandX = null;
                    lastHandY = null;
                }
            } else {
                lastHandX = null;
                lastHandY = null;
            }
        });

        const camera = new Camera(videoContainer, {
            onFrame: async () => {
                await hands.send({image: videoContainer});
            },
            width: 1280,
            height: 720
        });
        
        camera.start().catch(err => {
            console.error("Camera error:", err);
            const mirrorInput = document.getElementById('mirror-input');
            if(!mirrorInput.value) {
                mirrorInput.placeholder = "Please allow camera access! 💖\n" + mirrorInput.placeholder;
            }
        });
    } catch(err) {
        console.error("Tracking Error: ", err);
    }
}

initHandTracking();
