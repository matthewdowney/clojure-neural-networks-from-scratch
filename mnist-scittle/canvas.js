// Canvas variables
let WIDTH, HEIGHT;
const PIXEL_COUNT = 784;

let canvas, ctx;
let pixels = Array(PIXEL_COUNT).fill(0.0);
let cropped = Array(PIXEL_COUNT).fill(0.0);
let drawOn = true;
let mpushed = false;
let {xpos, ypos} = {xpos: -1, ypos: -1};

// Handlers for mouse events
const mouseDown = e => {
    mpushed = true;
    const x2 = (e.pageX - canvas.offsetLeft)/WIDTH;
    const y2 = (e.pageY - canvas.offsetTop)/HEIGHT;
    drawPixels(x2, y2, pixels);
    drawOn = true;
    e.preventDefault();
};

const mouseMove = e => {
    xpos = e.pageX - canvas.offsetLeft;
    ypos = e.pageY - canvas.offsetTop;
    if (mpushed){
        const x2 = xpos/WIDTH;
        const y2 = ypos/HEIGHT;
        drawPixels(x2, y2, pixels);
    }
    drawOn = true;
    e.preventDefault();
};

const mouseUpOrOut = e => {
    mpushed = false;
    drawOn = true;
    e.preventDefault();
};

// Helper function to draw pixels in a 28x28 square
const drawPixels = (x2, y2, pixelArray) => {
    for (let i=-1;i<2;i+=2)
        for (let j=-1;j<2;j+=2){
            const x = Math.round(x2 * 28 + i*0.5 - 0.5);
            const y = Math.round(y2 * 28 + j*0.5 - 0.5);
            if (x>=0 && x<28 && y>=0 && y<28)
                pixelArray[y*28+x] = 1;
        }
};

// Function to draw squares in the canvas
const square = (x,y,c) => {
    ctx.fillStyle = c === 0 ? '#FFFFFF' : '#000000';
    const xinc = WIDTH/28;
    const yinc = HEIGHT/28;
    ctx.fillRect(x*xinc,y*yinc,xinc,yinc);
};

// Function to clear canvas
const clear = () => {
    ctx.clearRect(0,0,WIDTH,HEIGHT);
    ctx.font="24px Arial";
    ctx.fillStyle="#BBBBBB";
    ctx.fillText("Draw a digit 0 to 9",50,100);
};

const draw = () => {
    if (!drawOn) return;
    clear();
    const draw = pixels.some(pixel => pixel !== 0);
    for (let i = 0; i < 28; i++)
        for (let j = 0; j < 28; j++)
            if (draw) square(i,j,pixels[j*28+i]);
    ctx.strokeStyle = '#999999';
    if (xpos !== -1 && ypos !== -1)
        ctx.strokeRect(xpos-12.5,ypos-12.5,25,25);
    drawOn = false;
};

// Crop and center the drawn digit in 20x20 pixels
function crop() {
    const norm = 20.0;
    const size = 28;
    let left = 0, right = size, top = 0, bottom = size;
    cropped.fill(0.0);

    // Get left, right, top, bottom boundaries
    [left, right, top, bottom] = ['left', 'right', 'top', 'bottom'].map((direction) => {
        const isVertical = direction === 'top' || direction === 'bottom';
        const start = direction === 'left' || direction === 'top' ? 0 : size - 1;
        const end = direction === 'left' || direction === 'top' ? size : -1;
        const step = direction === 'left' || direction === 'top' ? 1 : -1;

        for (let index = start; index !== end; index += step) {
            for (let subIndex = 0; subIndex < size; subIndex++) {
                const pixelIndex = isVertical ? size * subIndex + index : size * index + subIndex;
                if (pixels[pixelIndex] !== 0) {
                    return index;
                }
            }
        }
    });

    const width = bottom - top + 1;
    const height = right - left + 1;
    const scale = Math.min(norm / height, norm / width);
    const scaledWidth = Math.floor(width * scale);
    const scaledHeight = Math.floor(height * scale);
    const xOffset = Math.round((size - scaledWidth) / 2);
    const yOffset = Math.round((size - scaledHeight) / 2);

    for (let i = 0; i < scaledHeight; i++) {
        for (let j = 0; j < scaledWidth; j++) {
            const x = Math.floor(i / scale) + left;
            const y = Math.floor(j / scale) + top;
            cropped[size * (i + yOffset) + j + xOffset] = pixels[size * x + y];
        }
    }
}

// Initialization function
const init = () => {
    // initialize canvas
    canvas = document.getElementById("digitCanvas");
    WIDTH = canvas.width;
    HEIGHT = canvas.height;
    ctx = canvas.getContext("2d");
    canvas.addEventListener("contextmenu", e => {e.preventDefault(); return false;});
    canvas.addEventListener("mousedown", mouseDown);
    canvas.addEventListener("mouseup", mouseUpOrOut);
    canvas.addEventListener("mouseout", mouseUpOrOut);
    canvas.addEventListener("mousemove", mouseMove);

    // continually redraw canvases
    setInterval(draw, 25);
};

// On window load, initialize the app
window.onload = init;

// Expose needed functions to window for global access
window.doCrop = crop;
window.doReset = () => {
    pixels.fill(0);
    cropped.fill(0);
    clear();
}
window.croppedPixels = cropped;
