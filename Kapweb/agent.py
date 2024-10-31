from Kapweb.audio import Xtts
from Kapweb.image import SDXL,Ocr
from Kapweb.text import Helsinki
from Kapweb.download import Download,YtDownload
from os import path
from slugify import slugify

current_file = path.realpath(__file__)

langues_codes = {
    "fran\u00e7ais": "fr",
    "francais": "fr",
    "anglais": "en",
    "espagnol": "es",
    "allemand": "de",
    "italien": "it",
}
#crée une image d'un chien sur un vélo
class Agent:
    def image(self,result):
        images = SDXL(result['prompt'],True,1,"static")
        return 'markdown:<div style="text-align:center">'+result['prompt']+'<br/><img src="./app/static/'+images[0]+'" ></div>'
    def trad(self,result):
        if langues_codes[result['from']]:
            result['from'] = langues_codes[result['from']]
        if langues_codes[result['to']]:
            result['to'] = langues_codes[result['to']]
        return Helsinki(result['from'],result['to'],result['prompt'])
    def stt(self,result):
        prompt = result['prompt']
        soundfile = Xtts(prompt,"tts_models/multilingual/multi-dataset/xtts_v2",result['voice'],  f'{slugify(prompt)}', 'static'   )
        return 'markdown:<div style="text-align:center">'+prompt+'<br/><audio controls><source src="./app/static/'+soundfile+'" type="audio/wav">Your browser does not support the audio element.</audio></div>'
    def download(self,result):
        response = Download(result['url'],'static')
        return 'markdown:<div style="text-align:center"><a href="./app/static/'+response+'" target="_blank">'+result['url']+'</a></div>'
    def ytdownload(self,result):
        response = YtDownload(result['url'],'static')
        return 'markdown:<div style="text-align:center">'+result['url']+'<br/><video controls width="300"><source src="./app/static/'+response+'" type="video/mp4" /><a href="/media/cc0-videos/flower.mp4">MP4</a></video></div>'
    def ffmpeg(self,result):
        print(result)
        return "oki"
    def Ocr(self,result):
        response = Ocr(result['file'],'static')
        return response

    