import email
import imaplib
import logging
import os

from email.header import decode_header
from datetime import datetime, timedelta
import progressbar
from langchain.schema.document import Document

def retreiveEmail():
    email_user = "damien@kapweb.com"
    email_pass = "x5zf8WVt@kWBQ"
    print('get mails')
    mail = imaplib.IMAP4_SSL("ns2.kapweb.com")
    mail.login(email_user, email_pass)

    mail.select()

    date_since = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
    result, message_ids = mail.search(None, '(SINCE "{}")'.format(date_since))

    # cette partie sera peut-être a adapté selon le mail provider que vous utilisez
    if result != 'OK':
        logging.error('Bad response from third-party')
        exit(1)
        
    # extraire les identifiants de message de la réponse du provider
    message_ids = message_ids[0].split()

    # et ce tableau contiendra l'intégralité du corps de nos éléments
    docs = []
    count = 0

    for message_id in message_ids:
        count += 1
        result, message_data = mail.fetch(message_id, "(RFC822)")
        if result != 'OK':
            # on ignore en cas d'erreur, pour simplifier l'exemple
            continue
        raw_email = message_data[0][1]
        msg = email.message_from_bytes(raw_email)
        subject, encoding = decode_header(msg["Subject"])[0]
        # on décode le sujet dans le cas où ce n'est pas une simple chaîne de caractères
        if isinstance(subject, bytes):
            try:
                subject = subject.decode(encoding if encoding is not None else "utf-8")
            except LookupError:
                # le sujet est invalide, on ignore pour simplifier l'exemple à nouveau
                continue

        # ajout du sujet et des métadonnées dans le contenu
        content = f"""
        Subject: {subject}
        From: {msg["From"]}
        Date: {msg["Date"]}
        ===================
        """
        # décodage de chaque partie de l'email si celui-ci est multipart, ou
        # simplement de son contenu s'il est singlepart
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        content += body.decode("utf-8")
            else:
                body = msg.get_payload(decode=True)
                content += body.decode("utf-8")

        except UnicodeDecodeError:
            # on drop les emails invalides pour simplifier l'exemple
            continue

        # ajout du document à la liste en utilisant la classe `Document` de Langchain
        docs.extend([Document(page_content=content, metadata={
            "source": message_id,
            "subject": subject
        })])

        # finalement, on met à jour la barre de progression pour une meilleure expérience utilisateur

    mail.logout()   
    return docs