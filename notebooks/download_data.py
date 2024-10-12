import gdown

def download_file(file_id, output_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_name, quiet=False)

# Example usage:
# file_id1 = '1KU0EtfjOTVFyoovF2Z7MguHzZm_R_eGu'
# file_id2 = '14ZLT_HqmrQVOfG0qtz-u2PbtLdo7Ia9x'
# file_id3 = '1i7Bf2S5iSv_D2K77P6834jdY4jX_KlC5'
file_id4 = '1M9Q0gJfU-5KMuCgQFyONZRSBFkpO0lzc'
# download_file(file_id1, 'fineweb_post.zip')
# download_file(file_id2, 'fineweb_pre.zip')
# download_file(file_id3, 'fineweb_blank.zip')
download_file(file_id4, 'fineweb_none.zip')