"""
@author: mezdahun
@description: Helper functions to upload files to Google Drive automatically
"""
# [START drive_quickstart]
from __future__ import print_function
import os.path
from visualswarm.contrib import monitoring, simulation
from googleapiclient.discovery import build
import googleapiclient.http
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
import logging
import zipfile
import random
import string

if not simulation.ENABLE_SIMULATION:
    # setup logging
    import os

    ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
    logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
    logger.setLevel(monitoring.LOG_LEVEL)
else:
    logger = logging.getLogger('visualswarm.app_simulation')  # pragma: simulation no cover

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']
CURRDIR, _ = os.path.split(os.path.abspath(__file__))


def ensure_tokens():
    """
    First time authentication of agent with Google OAuth according to credentials file. Only authenticates
    with the account visualswarm.scioi@gmail.com
    """
    creds = None
    if monitoring.CLOUD_STORAGE_AUTH_MODE == 'OAuth2':
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        token_path = os.path.join(CURRDIR, 'token.json')
        cred_path = os.path.join(CURRDIR, 'credentials.json')
        if os.path.exists(token_path):
            logger.info('OAuth tokens have been found. Using them...')
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                logger.info('No token has been found, opening OAuth in Browser. Please open the link in a private tab if'
                            'you experience problems with login. '
                            'The only account allowed to login is visualswarm.scioi@gmail.com')
                flow = InstalledAppFlow.from_client_secrets_file(
                    cred_path, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
    elif monitoring.CLOUD_STORAGE_AUTH_MODE == 'ServiceAccount':
        cred_path = os.path.join(CURRDIR, 'service_key.json')
        creds = service_account.Credentials.from_service_account_file(cred_path)
    else:
        logger.error('Unknown method was chosen for Google Auth')

    service = build('drive', 'v3', credentials=creds)

    # Call the Drive v3 API
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    results.get('files', [])

    logger.info('Successful authentication! Token is valid!')
    return service


def upload_vision_videos(videos_folder=None):
    if videos_folder is None:
        videos_folder = monitoring.SAVED_VIDEO_FOLDER

    if os.path.isdir(videos_folder):
        drive_service = ensure_tokens()
        mimetype = 'video/mp4'

        for filename in os.listdir(videos_folder):
            if filename.endswith(".mp4"):
                filename = os.path.join(videos_folder, filename)
                name_parts = os.path.split(filename)[1].split('.')[0].split('_')
                logger.info(name_parts)
                video_timestamp, exp_id, robot_name = name_parts

                media_body = googleapiclient.http.MediaFileUpload(
                    filename,
                    mimetype=mimetype,
                    resumable=False
                )
                # The body contains the metadata for the file.
                body = {
                    'name': os.path.split(filename)[1],
                    'title': os.path.split(filename)[1],
                    'description': f"Experiment with ID: {exp_id} "
                                   f"Robot ID: {robot_name} "
                                   f"Started @ {video_timestamp}",
                }

                # Perform the request and print the result.
                new_file = drive_service.files().create(
                    body=body, media_body=media_body).execute()

                if monitoring.CLOUD_STORAGE_AUTH_MODE == 'ServiceAccount':
                    cloudPermissions = drive_service.permissions().create(fileId=new_file['id'],
                                                                          body={'type': 'user',
                                                                                'role': 'owner',
                                                                                 'emailAddress': 'visualswarm.scioi@gmail.com'}).execute()

                logger.info(f"\nFile created, id@drive: {new_file.get('id')}, local file: {os.path.split(filename)[1]}")
                logger.info("Deleting local copy after successful upload...")
                os.remove(filename)
                logger.info("Local copy deleted.\n")

    else:
        logger.warning('The passed video library does not exist. Will skip Google Drive Upload...')


def upload_statevars(videos_folder=None):
    if videos_folder is None:
        videos_folder = monitoring.SAVED_VIDEO_FOLDER

    if os.path.isdir(videos_folder):
        drive_service = ensure_tokens()
        mimetype = 'application/octet-stream'

        for filename in os.listdir(videos_folder):
            if filename.endswith(".npy"):
                filename = os.path.join(videos_folder, filename)
                name_parts = os.path.split(filename)[1].split('.')[0].split('_')
                logger.info(name_parts)
                video_timestamp, exp_id, robot_name, _ = name_parts

                media_body = googleapiclient.http.MediaFileUpload(
                    filename,
                    mimetype=mimetype,
                    resumable=False
                )
                # The body contains the metadata for the file.
                body = {
                    'name': os.path.split(filename)[1],
                    'title': os.path.split(filename)[1],
                    'description': f"Experiment with ID: {exp_id} "
                                   f"Robot ID: {robot_name} "
                                   f"Started @ {video_timestamp}",
                }

                # Perform the request and print the result.
                new_file = drive_service.files().create(
                    body=body, media_body=media_body).execute()

                if monitoring.CLOUD_STORAGE_AUTH_MODE == 'ServiceAccount':
                    cloudPermissions = drive_service.permissions().create(fileId=new_file['id'],
                                                                          body={'type': 'user',
                                                                                'role': 'owner',
                                                                                 'emailAddress': 'visualswarm.scioi@gmail.com'}).execute()

                logger.info(f"\nFile created, id@drive: {new_file.get('id')}, local file: {os.path.split(filename)[1]}")
                logger.info("Deleting local copy after successful upload...")
                os.remove(filename)
                logger.info("Local copy deleted.\n")

    else:
        logger.warning('The passed video library does not exist. Will skip Google Drive Upload...')

def zipdir(path, ziph):
    """all credits to: https://stackoverflow.com/a/1855118"""
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

def zipupload_CNN_training_data(training_data_folder=None):
    # zipping png files in folder
    videos_folder = monitoring.SAVED_VIDEO_FOLDER
    if training_data_folder is None:
        training_data_folder = os.path.join(videos_folder, 'training_data')
    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    zip_filename = f'CNNTD_{token}.zip'
    zip_path = os.path.join(videos_folder, zip_filename)
    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    zipdir(training_data_folder, zipf)
    zipf.close()

    logger.info(f'is zip file {zip_path} : {os.path.isfile(zip_path)}')

    drive_service = ensure_tokens()
    # uploading created zipfile
    media_body = googleapiclient.http.MediaFileUpload(
        zip_path,
        mimetype='application/octet-stream',
        resumable=False
    )

    # The body contains the metadata for the file.
    body = {
        'name': zip_filename,
        'title': zip_filename,
        'description': "Collected training data to finetune CNN based object detector"
    }

    # Perform the request and print the result.
    new_file = drive_service.files().create(
        body=body, media_body=media_body).execute()

    if monitoring.CLOUD_STORAGE_AUTH_MODE == 'ServiceAccount':
        cloudPermissions = drive_service.permissions().create(fileId=new_file['id'],
                                                              body={'type': 'user',
                                                                    'role': 'owner',
                                                                    'emailAddress': 'visualswarm.scioi@gmail.com'}).execute()

    logger.info(f"\nFile created, id@drive: {new_file.get('id')}, local file: {zip_filename}")
    logger.info("Deleting local copy after successful upload...")
    # os.remove(filename)
    # logger.info("Local copy deleted.\n")
