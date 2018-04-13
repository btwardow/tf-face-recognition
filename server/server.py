import detection
from PIL import Image
from flask import Flask, request, Response

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')  # Put any other methods you need here
    return response


@app.route('/detect')
def index():
    return Response(open('./static/detect.html').read(), mimetype="text/html")


@app.route('/image', methods=['POST'])
def image():
    try:
        image_stream = request.files['image']  # get the image
        image = Image.open(image_stream)

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        # finally run the image through tensor flow object detection`
        objects = detection.get_faces(image, threshold)
        return objects

    except Exception as e:
        import traceback
        traceback.print_exc()
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', ssl_context='adhoc')
    app.run(debug=True, host='0.0.0.0')
