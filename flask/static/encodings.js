
function toRunLengthEncoding(pixelArr) {
    // compress pixel data (taking advantage of it only having two values, black or transparent)
    // ie. instead of having [0, 0, 0, 0, 1, 1, 0, 0, 0], use the form [4, 2, 3]
    // as in: 4 transparent pixels followed by 2 black pixels followed by 3 white pixels
    // more info on the run-length encoding: https://en.wikipedia.org/wiki/Run-length_encoding

    var pixelArrCounts = []
    var onBlack = false;
    if (pixelArr[3] > 0) {
        // starting on a transparent pixel
        // (starting with "0 transparent pixels" just flips to black)
        pixelArrCounts.push(0)
        onBlack = true;
    }

    // count of black/transparent pixels in a row
    var count = 0;
    // canvas data has 4 components (RGBA) per pixel, so take steps of 4
    // to read only alpha values (check: transparent or not transparent pixel)
    for (var i = 3; i < pixelArr.length; i += 4) {
        count++;

        if ((pixelArr[i] > 0) != onBlack) {
            // change in pixel colour from the prevoius one, record count and flip to white/black
            pixelArrCounts.push(count);
            onBlack = !onBlack;
            count = 0;
        }
    }

    if(count > 0) {
        // still pixel counts to be recorded after loop
        pixelArrCounts.push(count);
    }

    return pixelArrCounts;
}

function toNumberArrayEncoding(numbersArr) {
    // converts an array of numbers into an encoding of bytes, where each byte
    // is two characters of the array as a string
    // (4 bits needed to represent the numbers 0-9 or a comma)

    // create number array as string
    var jsonArr = JSON.stringify(numbersArr);
    // get half length (ignoring the brackets at either side)
    // this is how many bytes we'll need to encode the  number array
    var len = (jsonArr.length - 2) / 2;
    if (jsonArr.length % 2 == 1) {
        // odd number of characters, need the extra byte
        // (since integer division is floored)
        len += 2;
    }

    // create byte buffer
    var buffer = new Uint8Array(len);
    var bufferCounter = 0;

    // ignoring the brackets on either side of the string
    for (var i = 1; i < jsonArr.length - 1; i += 2) {
        var nextByte = 0;

        // "," can just encode to 0 so we only have to change the byte value
        // if it's a number being encoded
        if (jsonArr.charAt(i) != ",") {
            // encode the number in the byte's leftmost 4 bits
            // (adding 1 since the comma's encoded value is already 0)
            nextByte = (jsonArr.charCodeAt(i) - 48 + 1) << 4;
        }

        if (i == jsonArr.length - 2) {
            // last character, must pad right bits with a comma and add in
            // two redundant zeroes as the next byte
            buffer[bufferCounter] = nextByte;
            buffer[bufferCounter + 1] = (1 << 4) | 1;
            break;
        }
        else if (jsonArr.charAt(i + 1) != ",") {
            // use a bitwise OR to add this number's encoding to the
            // rightmost 4 bits of the byte
            nextByte = nextByte | (jsonArr.charCodeAt(i + 1) - 48 + 1);
        }

        buffer[bufferCounter++] = nextByte;
    }

    return buffer;
}

// taken from https://gist.github.com/jonleighton/958841
// converts an ArrayBuffer of bytes into Base64
function base64ArrayBuffer(arrayBuffer) {
    var base64    = ''
    var encodings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    // may as well not create a new buffer for this, since I don't need the buffer afterwards
    // var bytes         = new Uint8Array(arrayBuffer)
    var bytes         = arrayBuffer
    var byteLength    = bytes.byteLength
    var byteRemainder = byteLength % 3
    var mainLength    = byteLength - byteRemainder

    var a, b, c, d
    var chunk

    // Main loop deals with bytes in chunks of 3
    for (var i = 0; i < mainLength; i = i + 3) {
        // Combine the three bytes into a single integer
        chunk = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2]

        // Use bitmasks to extract 6-bit segments from the triplet
        a = (chunk & 16515072) >> 18 // 16515072 = (2^6 - 1) << 18
        b = (chunk & 258048)   >> 12 // 258048   = (2^6 - 1) << 12
        c = (chunk & 4032)     >>  6 // 4032     = (2^6 - 1) << 6
        d = chunk & 63               // 63       = 2^6 - 1

        // Convert the raw binary segments to the appropriate ASCII encoding
        base64 += encodings[a] + encodings[b] + encodings[c] + encodings[d]
    }

    // Deal with the remaining bytes and padding
    if (byteRemainder == 1) {
        chunk = bytes[mainLength]

        a = (chunk & 252) >> 2 // 252 = (2^6 - 1) << 2

        // Set the 4 least significant bits to zero
        b = (chunk & 3)   << 4 // 3   = 2^2 - 1

        base64 += encodings[a] + encodings[b] + '=='
    } else if (byteRemainder == 2) {
        chunk = (bytes[mainLength] << 8) | bytes[mainLength + 1]

        a = (chunk & 64512) >> 10 // 64512 = (2^6 - 1) << 10
        b = (chunk & 1008)  >>  4 // 1008  = (2^6 - 1) << 4

        // Set the 2 least significant bits to zero
        c = (chunk & 15)    <<  2 // 15    = 2^4 - 1

        base64 += encodings[a] + encodings[b] + encodings[c] + '='
    }
    
    return base64
}