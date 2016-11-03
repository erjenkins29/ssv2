from email.Header import Header
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart
import smtplib, datetime, sys

def mailtx(msg_text,
           attachments=None,
           mailto='teddy.wen@earlydata.com',
           ccto = None):
    ### if "attachments" has more than one file, it should be a list;
    debuglevel = 0

    #user infomation
    server = smtplib.SMTP()
    server.set_debuglevel(debuglevel)
    server.connect('smtp.exmail.qq.com')
    server.login("email-Address","Passwd")
    
    #attach the file
    msg = MIMEMultipart() #means the email will include mulitiple parts.
    if attachments:
        for attachment in attachments:
            att = MIMEText(open(attachment, 'rb').read(), 'base64', 'utf-8')
            att["Content-Type"] = 'application/octet-stream'
            att["Content-Disposition"] = 'attachment; filename=%s' %(os.path.basename(attachment))
            msg.attach(att)
    
    #sending infomation
    msg['to'] = mailto
    msg['from'] = 'teddy.wen@earlydata.com'
    if ccto:
        msg['CC'] = ccto
    msg['subject'] = Header('ss project (' + str(datetime.date.today()) + ')','utf-8')
    
    #msg body
    message_text = msg_text + "\n\nThis was sent automatically via python\n\n"\
+sys.version
        
    msg.attach(MIMEText(message_text, 'plain'))
    server.sendmail(msg['from'], msg['to'], msg.as_string())
    server.quit()


##mailtx(msg_text="hi,hello, how are you?",
       ##attachments=['latestdatainfo.txt','predictions.py','test.ipynb'])

