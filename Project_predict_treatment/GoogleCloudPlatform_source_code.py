source code::
import requests
import pickle
from google.cloud import storage
import numpy as np

def process_ml_query(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('dissertation-project-work')
    blob_classifier = bucket.blob('predictTreatment.pickle')
    blob_classifier.download_to_filename('/tmp/predictTreatment.pickle')
    serverless_model = pickle.load(open('/tmp/predictTreatment.pickle', 'rb'))
    attributesKeys = ["AffectedArea", "severity", "Age", "Tumortype", "Size"]
    attributeValueList = []

    for attribute in attributesKeys:
        attributeValueList.append(request_json[attribute])

    feature_data = [attributeValueList]
    prediction = serverless_model.predict(feature_data)
    return "Predicted Class : {}".format(str(prediction))


# Please keep this under main.py file of google cloud functions
