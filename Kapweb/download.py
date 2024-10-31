import requests
from os import path
import os
from slugify import slugify
from urllib.parse import urlparse
from pytube import YouTube

current_file = path.realpath(__file__)

def Download(url,output_dir):
    response = requests.get(url)
    headers = response.headers
    mime_type = headers.get('Content-Type')
    
    parsed_url = urlparse(url)
    filename    = parsed_url.netloc+"_" 
    
    extension = mime_type.split('/')[-1].split(';')[0]
    if '+' in extension:
      extension = extension.split('+')[0]
      
    static_filename = f'{slugify(url)}.{extension}'
    filename = path.join(path.dirname(current_file),"..", output_dir,static_filename)
    with open(filename, "wb") as f:
        f.write(response.content)
    return static_filename
        
def YtDownload(url,output_dir):
    static_filename = f'{slugify(url)}.mp4'
    output_dir      = path.join(path.dirname(current_file),"..", output_dir+'/')
    if not os.path.exists(output_dir+static_filename):
      print(output_dir+static_filename)
      yt = YouTube('https://www.youtube.com/'+url)
      stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
      stream.download(output_path=output_dir, filename=static_filename)
      
    return static_filename