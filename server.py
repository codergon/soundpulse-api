from io import BytesIO
import json

from pathlib import Path

from sanic import Sanic
from sanic.request import Request
from sanic.request.form import File
from sanic.response import json, JSONResponse

import numpy as np
import librosa as lb
import tensorflow as tf

application = Sanic("soundpulse-api")


@application.before_server_start
def load_model(app, _):
    model_path = (Path(__file__).resolve().parent / "models" / "model.tflite").resolve()
    app.ctx.interpreter = tf.lite.Interpreter(model_path=str(model_path))
    app.ctx.interpreter.allocate_tensors()


@application.before_server_start
def load_model_labels(app, _):
    labels_mapping = {}
    labels_path = (Path(__file__).resolve().parent / "models" / "labels.txt").resolve()
    with labels_path.open("r") as labels_file:
        for line in labels_file:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                index, label = parts
                labels_mapping[int(index)] = label

    app.ctx.labels_mapping = labels_mapping


@application.get("/")
async def index(request: Request) -> JSONResponse:
    return json(None)


@application.post("/predict")
def predict(request: Request) -> JSONResponse:
    audio_file: File | None = request.files.get("audio")
    if audio_file is None:
        return json({"message": "`audio` is missing from form data"}, status=422)

    try:
        audio, _ = lb.load(BytesIO(audio_file.body), sr=None)
        audio, _ = lb.effects.trim(audio)
    except Exception as e:
        return json(
            {"message": "Unable to load audio file. Provide a supported format"},
            status=400,
        )

    input_details = application.ctx.interpreter.get_input_details()
    expected_audio_length = input_details[0]["shape"][1]
    if len(audio) > expected_audio_length:
        audio = audio[:expected_audio_length]
    else:
        audio = np.pad(audio, (0, expected_audio_length - len(audio)), "constant")

    audio_input = audio.reshape(1, -1).astype(np.float32)

    application.ctx.interpreter.set_tensor(input_details[0]["index"], audio_input)
    application.ctx.interpreter.invoke()

    output_details = application.ctx.interpreter.get_output_details()
    output_data = application.ctx.interpreter.get_tensor(output_details[0]["index"])

    parsed_output = map_labels_to_values(
        values=output_data.tolist()[0],
        labels_mapping=application.ctx.labels_mapping,
    )
    return json({"data": parsed_output})


def map_labels_to_values(labels_mapping, values):
    result = {}
    for index, label in labels_mapping.items():
        result[label] = values[index]

    return result