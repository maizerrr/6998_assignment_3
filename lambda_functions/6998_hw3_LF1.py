import json
import logging
import boto3

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
            'body': e
        }

    email = None
    with boto3.resource('s3') as s3:
        email = s3.Object(bucket, filename)
        email.get()['Body'].read().decode('utf-8')
    if email is None:
        logger.error("Failed to retrieve email content from s3")
    else: 
        logger.info(email)
    
    # connect to sagemaker endpoint
    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName='sms-spam-classifier-mxnet-2022-12-04-19-08-20-432',
        Body=email
    )

    logger.info(response)


    return {
        'statusCode': 200,
        'body': None
    }