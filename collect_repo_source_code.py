import config as cfg
import requests
import logging
import re
from lang_ext_mapping import lang_ext_mapping
from db_manager import repositories, update_status, close
from data_collection import get_json_from_url, check_rate_limit

headers = {'Authorization': 'token %s' % cfg.oauth_token}

def get_file_contents(url):
    file_contents = get_json_from_url(url)
    if file_contents == -1:
        logging.error('get_file_contents got -1 when requesting from %s' % url)
        return -1
    if type(file_contents) is not list:
        logging.error('get_file_contents got wrong data structure when requesting from %s' % url)
        return -1
    return file_contents

def get_url_content(url):
    try:
        req = requests.get(url, headers=headers)
        content = req.text
    except:
        logging.error('An error occured when requesting content from this url: %s' % url)
        return -1
    if content.find('API rate limit exceeded for user') != -1:
        return check_rate_limit(get_url_content)(url)
    return content.strip()

def get_lang_ext_mappings(languages):
    exts = []
    for lang in languages.keys():
        if lang in lang_ext_mapping:
            exts += lang_ext_mapping[lang]
    return exts

def escape_file_name(name):
    new_name = re.sub('^\$', '%' + format(ord('$'), "x"), name)
    new_name = re.sub('\.', '%' + format(ord('.'), "x"), new_name)
    return new_name

def update_repo_files(repo_id, url, programming_languages):
    file_contents = get_file_contents(url)
    if file_contents == -1:
        return
    for file_content in file_contents:
        if file_content['type'] == 'dir':
            update_repo_files(repo_id, file_content['url'], programming_languages)
        elif file_content['type'] == 'file':
            file_ext_match = re.match('(.*)\.md$', file_content['name'], re.IGNORECASE)
            if file_ext_match is not None:
                content = get_url_content(file_content['download_url'])
                if content != -1 and content != '':
                    file_name = escape_file_name(file_ext_match.group(1))
                    # Size must be between 0 and 16793600(16MB)
                    if len(content) > 1679360:
                        content = content[: 1679360]
                    try:
                        repositories.update_one({'_id': repo_id}, {'$set': {'read_me.%s' % file_name: content}})
                    except Exception as e:
                        update_status.update_one({'_id': repo_id}, {'$set': {'error.%s' % file_name: str(e)}})
                        logging.error('An error occured when processing this url: %s, error message: %s' % (url, str(e)))
                        break
                continue
            file_ext_match = re.match('.*(\..*)$', file_content['name'], re.IGNORECASE)
            if file_ext_match is not None and file_ext_match.group(1) in programming_languages:
                content = get_url_content(file_content['download_url'])
                if content != -1 and content != '':
                    file_name = escape_file_name(file_content['name'])
                    # Size must be between 0 and 16793600(16MB)
                    if len(content) > 1679360:
                        content = content[: 1679360]
                    try:
                        repositories.update_one({'_id': repo_id}, {'$set': {'source_code.%s' % file_name: content}})
                    except Exception as e:
                        update_status.update_one({'_id': repo_id}, {'$set': {'error.%s' % file_name: str(e)}})
                        logging.error('An error occured when processing this url: %s, error message: %s' % (url, str(e)))
                        break
                continue
    return

def update_repository_data():
    # select repos from the database
    repos_count = repositories.count_documents({'$and': [{'disabled': False}, {'archived': False}]})
    all_repos = repositories.find({'$and': [{'disabled': False}, {'archived': False}]})
    progress_counter = 1

    for repo in all_repos:
        # archived or disabled repositorues should not be considered
        if repo['archived'] is True or repo['disabled'] is True:
            continue

        repo_update_status = update_status.find_one({'_id': repo['_id']})
        if repo_update_status is not None and \
            ('updated' in repo_update_status or 'updating' in repo_update_status):
            progress_counter += 1
            continue

        logging.warning('Total number of repositories: %s, currently processing: %s' % (repos_count, progress_counter))

        update_status.insert_one({'_id': repo['_id'], 'updating': True})

        languages = get_lang_ext_mappings(repo['languages_detail'])
        # extract the most dominant one
        languages = languages[0:1]
        update_repo_files(repo['_id'], repo['contents_url'].format(**{'+path':''}), languages)
        progress_counter += 1

        update_status.update_one({'_id': repo['_id']}, {'$set': {'updated': True}, '$unset': {'updating': ''}})

    close()

if __name__ == "__main__":
    update_repository_data()