import json
import boto3
import logging
import email

from io import StringIO

from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def lambda_handler(event, context):
    
    logger.info(event)

    # validate event
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        filename = event["Records"][0]["s3"]["object"]["key"]
        t = event["Records"][0]["eventTime"]
    except Exception as e:
        logger.error("Exception encountered when parsing event")
        return {
            'statusCode': 500,
            'body': str(e)
        }

    # retrieve and parse email file
    s3 = boto3.resource('s3')
    file = s3.Object(bucket, filename)
    file = file.get()['Body'].read()
    file = email.message_from_bytes(file)
    if file.is_multipart():
        for _file in file.get_payload():
            msg = _file.get_payload()
            break
    else:
        msg = file.get_payload
    
    
    # sagemaker
    client = boto3.client('sagemaker-runtime')
    _msg = vectorize_sequences( one_hot_encode([msg], 9013), 9013 )
    
    io = StringIO()
    json.dump(_msg.tolist(), io)
    
    try:
        response = client.invoke_endpoint(
            EndpointName='sms-spam-classifier-mxnet-2022-12-04-19-08-20-432',
            Body=bytes(io.getvalue(), 'utf-8')
        )
        response = response['Body'].read().decode()
        response = json.loads( str(response) )
        label = "ham" if response['predicted_label'] == [[0.0]] else "spam"
        logger.info("Response body from model endpoint: {}".format(response))
    except Exception as e:
        logger.error("Unable to parse model response")
        return {
            'statusCode': 500,
            'body': str(e),
            'response': response
        }
        

    # send responding email back to sender
    ses = boto3.client('ses')
    CHARSET = "UTF-8"
    
    content = """
    We received your email sent at {} with the subject {}.
    
    Here is a 240 character sample of the email body: {}
    
    The email was categorized as {} with a {:.5}% confidence.
    """.format( t, file['subject'], msg[:240], label, response['predicted_probability'][0][0] * 100 )
    
    logger.info(content)
    
    res = ses.send_email(
        Source='admin@spam.maizer.pw',
        Destination={
            'ToAddresses': [file['from']],
        },
        Message={
            'Subject': {
                'Data': 'email recieved',
                'Charset': CHARSET
            },
            'Body': {
                'Text': {
                    'Data': content,
                    'Charset': CHARSET
                }
            }
        }
    )

    return {
        'statusCode': 200,
        'body': None
    }