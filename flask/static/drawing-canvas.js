// adapted from https://stackoverflow.com/a/8398189
var canvas, ctx;
// user currently drawing (mouse is down and inside the canvas)
var drawing = false;
var hasDrawn = false;
// current mouse position
var currX = 0,
    currY = 0;
// drawing colour and base line width
var drawColour = "black",
    baseLineWidth = 5;
var drawPoints = [];
var isMobile = false;

function init() {
    // from https://stackoverflow.com/a/3540295
    // detect if this is a touch device or not
    if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|ipad|iris|kindle|Android|Silk|lge |maemo|midp|mmp|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows (ce|phone)|xda|xiino/i.test(navigator.userAgent) 
        || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(navigator.userAgent.substr(0,4))) { 
        isMobile = true;
    }
    
    // initialise canvas
    canvas = document.getElementById('digit-canvas');
    ctx = canvas.getContext("2d");
    ctx.strokeStyle = drawColour;
    ctx.lineCap = "round";

    // register mouse event listeners
    canvas.addEventListener("mousedown", function (e) {
        onMouseEvent("mousedown", e);
    });
    canvas.addEventListener("mouseup", function (e) {
        onMouseEvent("mouseup", e);
    });
    canvas.addEventListener("mousemove", function (e) {
        onMouseEvent("mousemove", e);
    });
    canvas.addEventListener("mouseout", function (e) {
        onMouseEvent("mouseout", e);
    });

    // register event listeners for touch devices
    canvas.addEventListener("touchstart", function (e) {
        onMouseEvent("touchstart", e);
    });
    canvas.addEventListener("touchend", function (e) {
        onMouseEvent("touchend", e);
    });
    canvas.addEventListener("touchmove", function (e) {
        onMouseEvent("touchmove", e);
    });
    canvas.addEventListener("touchcancel", function (e) {
        onMouseEvent("touchcancel", e);
    });
}

function onMouseEvent(action, e) {
    if (isMobile) {
        // prevent back/forward swipes etc. in the browser on mobile devices
        e.preventDefault();
    }

    switch (action) {
        case "mousedown":
        case "touchstart":
            // update currX, currY from event vars
            updateCursorPos(e)

            // start drawing
            if (!hasDrawn) {
                // start a new list of drawing points
                drawPoints = [];
                ctx.lineWidth = baseLineWidth;
            }
            else {
                // user is continuing on their drawing, make sure this point
                // doesn't get drawn connected to the previous point
                drawPoints.push("up");
            }
            drawing = hasDrawn = true;
            ctx.moveTo(currX, currY);
            lineToPoint(currX, currY);

            drawPoints.push([currX, currY]);
            break;
        case "mouseup":
        case "touchend":
        case "mouseout":
            if (drawing) {
                drawing = false;

                // resize line width based on how big the digit the user drew was
                autoResizeLineWidth();

                // automatically classify canvas on mouse out/up
                classify();
            }
            break;
        case "mousemove":
        case "touchmove":
            if (drawing) {
                // update currX, currY from event vars
                updateCursorPos(e)
                // draw a line from where the mouse was to where it is now
                lineToPoint(currX, currY);

                drawPoints.push([currX, currY]);
            }
            break;
    }
}

function updateCursorPos(e) {
    // update current mouse position based on event vars
    // (different on mobile devices)
    if (isMobile) {
        currX = e.touches[0].clientX - canvas.getBoundingClientRect().left;
        currY = e.touches[0].clientY - canvas.getBoundingClientRect().top;
    }
    else {
        currX = e.clientX - canvas.getBoundingClientRect().left;
        currY = e.clientY - canvas.getBoundingClientRect().top;
    }
}

function lineToPoint(x, y) {
    // connect a rounded line between where the mouse was and where the mouse currently is
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function reset() {
    // clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // clear classification
    $("#classification").text("");
    $("#classify-time").text("");

    drawing = hasDrawn = false;
}

function recreateDrawing() {
    // redraw user draw digit on canvas from list of recorded points
    if (drawPoints.length == 0) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.moveTo(drawPoints[0][0], drawPoints[0][1]);

    for (var i = 0; i < drawPoints.length; i++) {
        var drawPoint = drawPoints[i];
        if (drawPoint == "up") {
            // user stopped drawing this line before continuing drawing, move to next point
            i++;
            ctx.moveTo(drawPoints[i][0], drawPoints[i][1]);
        }
        else {
            lineToPoint(drawPoint[0], drawPoint[1]);
        }
    }
}

function autoResizeLineWidth() {
    // automatically resize line width based on the size of the image the user drew
    var canvasSize = canvas.width;

    // find the bounding box around what the user drew
    var left = canvasSize;
    var right = 0;
    var top = canvasSize;
    var bottom = 0;

    var rawCanvasData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (var i = 3; i < rawCanvasData.length; i += 4) {
        if (rawCanvasData[i] > 0) {
            // compute pixel index (since there are 4 components per pixel)
            var index = (i - 3) / 4;
            // compute x & y values from index
            var x = index % canvasSize;
            var y = index / canvasSize;

            // update bounding box
            left = Math.min(x, left);
            right = Math.max(x, right);
            top = Math.min(y, top);
            bottom = Math.max(y, bottom);
        }
    }

    // compute new line width
    // (0.11 * width or height, whichever was bigger)
    var lineWidthFraction = 0.11;
    var newLineWith = Math.ceil(Math.max(right - left, bottom - top) * lineWidthFraction);
    // clamp between 1 to 15
    newLineWidth = Math.min(newLineWith, 15);
    newLineWidth = Math.max(newLineWith, 1);

    ctx.lineWidth = newLineWith;

    // redraw digit with new line width
    recreateDrawing();
}

function classify() {
    // classify canvas image digit
    var classifyStart = new Date().getTime();

    // load canvas data
    var rawCanvasData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var pixelArr = rawCanvasData.data;

    // encode pixel data using run-length encoding
    var runLengthEncoded = toRunLengthEncoding(pixelArr);

    // pass the data through another encoding, squashing each pair of
    // two characters into a single byte (since the array as a string only
    // has the values 0-9 and commas)
    var byteBufferEncoded = toNumberArrayEncoding(runLengthEncoded);

    // finally, encode the byte array as a Base64 string
    var base64Encoded = base64ArrayBuffer(byteBufferEncoded);

    // put together an object representing the encoded canvas data
    var encodedCanvas = {
        // assuming the canvas dimensions will always be square
        size: rawCanvasData.width,
        pixelData: base64Encoded
    }

    // make an AJAX POST request, sending up the canvas data
    // the resulting classification will be displayed to the user as text
    // more here: https://api.jquery.com/jQuery.post/
    $.post("/classify", encodedCanvas).done(function(result) {
        var timeTaken = new Date().getTime() - classifyStart;

        $("#classification").text(result);
        $("#classify-time").text("Классифицируется в" + timeTaken + " мс.");
    });
}
