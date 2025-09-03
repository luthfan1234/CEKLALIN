from twilio.rest import Client
import os

ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBER = os.environ.get('RECIPIENT_PHONE_NUMBER') # Nomor petugas

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_critical_alert(event_data: dict):
    try:
        message_body = (
            f"🚨 ALERT KECELAKAAN 🚨\n"
            f"Kamera: ID Stream {event_data['stream_id']}\n"
            f"Event: {event_data['event_type']} ({event_data['object_type']})\n"
            f"Lokasi: Detail - {event_data['details']}\n"
            f"Timestamp: {event_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        message = client.messages.create(
            from_=f'whatsapp:{TWILIO_PHONE_NUMBER}', # atau `from_=TWILIO_PHONE_NUMBER` untuk SMS
            body=message_body,
            to=f'whatsapp:{RECIPIENT_PHONE_NUMBER}' # atau `to=RECIPIENT_PHONE_NUMBER` untuk SMS
        )
        print(f"Notification sent successfully: {message.sid}")
    except Exception as e:
        print(f"Failed to send Twilio notification: {e}")