import config as cfg
from os import listdir
from os.path import join
import gzip
import json
import requests
from time import sleep
import datetime
import math
import logging
import pymongo
import re

# list gzip files which was downloaded from https://www.gharchive.org/
gz_json_files = listdir(cfg.data_path)
# GitHub user token which will be used to authorise the API requests hereinafter 
headers = {'Authorization': 'token %s' % cfg.oauth_token}

# No Sql database which will be used to store the users and projects data
client = pymongo.MongoClient(cfg.db_conn_str, serverSelectionTimeoutMS=5000)
db = client.github_project
users = db.users
repositories = db.repositories

logging.basicConfig(filename='data_collection.log', encoding='utf-8', level=logging.INFO)

def get_json_from_url(url: str):
    req = requests.get(url, headers=headers)
    try:
        req_json = req.json()
    except:
        logging.error('An error encountered when parsing json content from this url: %s' % url)
        return -1
    if 'message' in req_json:
        logging.error('This url returns an error: %s' % req_json['message'])
        return -1
    return req_json

# A decorator who checks the rate limit of a request.
# if a limit is reached then wait until the limit is lifted.
def check_rate_limit(func):
    def wrapper(*args, **kwargs):
        url = 'https://api.github.com/rate_limit'
        rate_limit = get_json_from_url(url)
        if rate_limit == -1:
            logging.error('Got -1 when requesting from %s' % url)
            return -1
        rate = rate_limit['rate']
        limit = rate['limit']
        used = rate['used']
        remaining = rate['remaining']
        reset = rate['reset']

        logging.info('Rate limit: %s, used: %s, remaining: %s, reset: %s' % (limit, used, remaining, reset))

        if remaining == 0:
            time_diff = reset - datetime.datetime.now().timestamp()
            time_diff = math.ceil(time_diff)
            logging.warn('Limit reached, sleep for %s seconds!' % time_diff)
            sleep(time_diff)
        
        return func(*args, **kwargs)
    return wrapper

@check_rate_limit
def get_user(user_url: str, add_repos=False):
    user_detail = get_json_from_url(user_url)
    if user_detail == -1:
        logging.error('Got -1 when requesting from %s' % user_url)
        return -1
    user_dict = {}
    user_dict['_id'] = user_detail['id']
    user_dict['user_name'] = user_detail['login']
    user_dict['avatar_url'] = user_detail['avatar_url']
    user_dict['gravatar_id'] = user_detail['gravatar_id']
    user_dict['url'] = user_detail['url']
    user_dict['html_url'] = user_detail['html_url']
    user_dict['followers_url'] = user_detail['followers_url']
    user_dict['following_url'] = user_detail['following_url']
    user_dict['gists_url'] = user_detail['gists_url']
    user_dict['starred_url'] = user_detail['starred_url']
    user_dict['subscriptions_url'] = user_detail['subscriptions_url']
    user_dict['organizations_url'] = user_detail['organizations_url']
    user_dict['repos_url'] = user_detail['repos_url']
    user_dict['events_url'] = user_detail['events_url']
    user_dict['received_events_url'] = user_detail['received_events_url']
    user_dict['user_type'] = user_detail['type']
    user_dict['site_admin'] = user_detail['site_admin']
    user_dict['name'] = user_detail['name']
    user_dict['company'] = user_detail['company']
    user_dict['blog'] = user_detail['blog']
    user_dict['location'] = user_detail['location']
    user_dict['email'] = user_detail['email']
    user_dict['hireable'] = user_detail['hireable']
    user_dict['bio'] = user_detail['bio']
    user_dict['twitter_username'] = user_detail['twitter_username']
    user_dict['public_repos'] = user_detail['public_repos']
    user_dict['public_gists'] = user_detail['public_gists']
    user_dict['followers'] = user_detail['followers']
    user_dict['following'] = user_detail['following']
    user_dict['created_at'] = datetime.datetime.strptime(user_detail['created_at'], '%Y-%m-%dT%H:%M:%SZ')
    user_dict['updated_at'] = datetime.datetime.strptime(user_detail['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
    user_dict['followings'] = get_following_list(user_dict['following_url'])
    user_dict['starred_repos'] = get_starred_repos(user_dict['starred_url'], add_repos)
    user_dict['subscriptions'] = get_subscriptions(user_dict['subscriptions_url'], add_repos)
    user_dict['repos'] = get_user_repos(user_dict['repos_url'], add_repos)

    return user_dict

@check_rate_limit
def get_repo(repo_url: str, add_users=False):
    repo_detail = get_json_from_url(repo_url)
    if repo_detail == -1:
        logging.error('Got -1 when requesting from %s' % repo_url)
        return -1
    repo_dict = {}
    repo_dict['_id'] = repo_detail['id']
    repo_dict['name'] = repo_detail['name']
    repo_dict['full_name'] = repo_detail['full_name']
    repo_dict['private'] = repo_detail['private']
    repo_dict['owner'] = repo_detail['owner']
    repo_dict['html_url'] = repo_detail['html_url']
    repo_dict['description'] = repo_detail['description']
    repo_dict['fork'] = repo_detail['fork']
    repo_dict['url'] = repo_detail['url']
    repo_dict['forks_url'] = repo_detail['forks_url']
    repo_dict['keys_url'] = repo_detail['keys_url']
    repo_dict['collaborators_url'] = repo_detail['collaborators_url']
    repo_dict['teams_url'] = repo_detail['teams_url']
    repo_dict['hooks_url'] = repo_detail['hooks_url']
    repo_dict['issue_events_url'] = repo_detail['issue_events_url']
    repo_dict['events_url'] = repo_detail['events_url']
    repo_dict['assignees_url'] = repo_detail['assignees_url']
    repo_dict['branches_url'] = repo_detail['branches_url']
    repo_dict['tags_url'] = repo_detail['tags_url']
    repo_dict['blobs_url'] = repo_detail['blobs_url']
    repo_dict['git_tags_url'] = repo_detail['git_tags_url']
    repo_dict['git_refs_url'] = repo_detail['git_refs_url']
    repo_dict['trees_url'] = repo_detail['trees_url']
    repo_dict['statuses_url'] = repo_detail['statuses_url']
    repo_dict['languages_url'] = repo_detail['languages_url']
    repo_dict['stargazers_url'] = repo_detail['stargazers_url']
    repo_dict['contributors_url'] = repo_detail['contributors_url']
    repo_dict['subscribers_url'] = repo_detail['subscribers_url']
    repo_dict['subscription_url'] = repo_detail['subscription_url']
    repo_dict['commits_url'] = repo_detail['commits_url']
    repo_dict['git_commits_url'] = repo_detail['git_commits_url']
    repo_dict['comments_url'] = repo_detail['comments_url']
    repo_dict['issue_comment_url'] = repo_detail['issue_comment_url']
    repo_dict['contents_url'] = repo_detail['contents_url']
    repo_dict['compare_url'] = repo_detail['compare_url']
    repo_dict['merges_url'] = repo_detail['merges_url']
    repo_dict['archive_url'] = repo_detail['archive_url']
    repo_dict['downloads_url'] = repo_detail['downloads_url']
    repo_dict['issues_url'] = repo_detail['issues_url']
    repo_dict['pulls_url'] = repo_detail['pulls_url']
    repo_dict['milestones_url'] = repo_detail['milestones_url']
    repo_dict['notifications_url'] = repo_detail['notifications_url']
    repo_dict['labels_url'] = repo_detail['labels_url']
    repo_dict['releases_url'] = repo_detail['releases_url']
    repo_dict['deployments_url'] = repo_detail['deployments_url']
    repo_dict['created_at'] = datetime.datetime.strptime(repo_detail['created_at'], '%Y-%m-%dT%H:%M:%SZ')
    repo_dict['updated_at'] = datetime.datetime.strptime(repo_detail['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
    repo_dict['pushed_at'] = None if repo_detail['pushed_at'] is None else datetime.datetime.strptime(repo_detail['pushed_at'], '%Y-%m-%dT%H:%M:%SZ')
    repo_dict['git_url'] = repo_detail['git_url']
    repo_dict['ssh_url'] = repo_detail['ssh_url']
    repo_dict['clone_url'] = repo_detail['clone_url']
    repo_dict['svn_url'] = repo_detail['svn_url']
    repo_dict['homepage'] = repo_detail['homepage']
    repo_dict['size'] = repo_detail['size']
    repo_dict['stargazers_count'] = repo_detail['stargazers_count']
    repo_dict['watchers_count'] = repo_detail['watchers_count']
    repo_dict['language'] = repo_detail['language']
    repo_dict['has_issues'] = repo_detail['has_issues']
    repo_dict['has_projects'] = repo_detail['has_projects']
    repo_dict['has_downloads'] = repo_detail['has_downloads']
    repo_dict['has_wiki'] = repo_detail['has_wiki']
    repo_dict['has_pages'] = repo_detail['has_pages']
    repo_dict['forks_count'] = repo_detail['forks_count']
    repo_dict['mirror_url'] = repo_detail['mirror_url']
    repo_dict['archived'] = repo_detail['archived']
    repo_dict['disabled'] = repo_detail['disabled']
    repo_dict['open_issues_count'] = repo_detail['open_issues_count']
    repo_dict['license'] = repo_detail['license']
    repo_dict['forks'] = repo_detail['forks']
    repo_dict['open_issues'] = repo_detail['open_issues']
    repo_dict['watchers'] = repo_detail['watchers']
    repo_dict['default_branch'] = repo_detail['default_branch']
    repo_dict['temp_clone_token'] = repo_detail['temp_clone_token']
    if 'parent' in repo_detail:
        repo_dict['parent'] = repo_detail['parent']
    if 'source' in repo_detail:
        repo_dict['source'] = repo_detail['source']
    repo_dict['network_count'] = repo_detail['network_count']
    repo_dict['subscribers_count'] = repo_detail['subscribers_count']
    repo_dict['languages_detail'] = get_repo_languages_detail(repo_detail['languages_url'])

    if add_users:
        add_star_gazers(repo_detail['stargazers_url'])
        add_contributors(repo_detail['contributors_url'])
        add_subscribers(repo_detail['subscribers_url'])
        add_repo_owner(repo_detail['owner']['url'])

    return repo_dict

@check_rate_limit
def get_following_list(following_url: str):
    following_list = list()
    req_url = following_url.format(**{'/other_user': ''})
    followings = get_json_from_url(req_url)
    if followings == -1:
        return [-1]
    for following in followings:
        id = following['id']
        following_list.append(id)
            
    return following_list

@check_rate_limit
def get_subscriptions(subscriptions_url: str, add_repos=False):
    subscription_list = list()
    subscriptions = get_json_from_url(subscriptions_url)
    if subscriptions == -1:
        return [-1]
    for subscription in subscriptions:
        id = subscription['id']
        subscription_list.append(id)
        if add_repos:
            if repositories.find_one({'_id': id}) is None:
                # add the repo to the database
                repo_url = subscription['url']
                repo_details = get_repo(repo_url)
                if repo_details != -1:
                    repositories.insert_one(repo_details)
    
    return subscription_list

@check_rate_limit
def get_starred_repos(starred_url: str, add_repos=False):
    starred_repos_list = list()
    req_url = starred_url.format(**{'/owner': '', '/repo': ''})
    starred_repos = get_json_from_url(req_url)
    if starred_repos == -1:
        return [-1]
    for repo in starred_repos:
        id = repo['id']
        starred_repos_list.append(id)
        if add_repos:
            if repositories.find_one({'_id': id}) is None:
                # add the repo to the database
                repo_url = repo['url']
                repo_details = get_repo(repo_url)
                if repo_details != -1:
                    repositories.insert_one(repo_details)
    
    return starred_repos_list

@check_rate_limit
def get_user_repos(repos_url: str, add_repos=False):
    repo_list = list()
    repos = get_json_from_url(repos_url)
    if repos == -1:
        return [-1]
    for repo in repos:
        id = repo['id']
        repo_list.append(id)
        if add_repos:
            if repositories.find_one({'_id': id}) is None:
                # add the repo to the database
                repo_url = repo['url']
                repo_details = get_repo(repo_url)
                if repo_details != -1:
                    repositories.insert_one(repo_details)
    
    return repo_list

@check_rate_limit
def get_repo_languages_detail(languages_url: str):
    languages = get_json_from_url(languages_url)
    if languages == -1:
        return -1
    # escaping reserved characters in mongodb
    languages_escaped = {}
    for language in languages:
        new_key = re.sub('^\$', '%' + format(ord('$'), "x"), language)
        new_key = re.sub('\.', '%' + format(ord('.'), "x"), new_key)
        languages_escaped[new_key] = languages[language]
    return languages_escaped

@check_rate_limit
def add_star_gazers(stargazers_url: str):
    stargazers = get_json_from_url(stargazers_url)
    if stargazers == -1:
        return
    for stargazer in stargazers:
        id = stargazer['id']
        user_url = stargazer['url']
        if users.find_one({'_id': id}) is None:
            user = get_user(user_url)
            if user != -1:
                users.insert_one(user)

@check_rate_limit
def add_contributors(contributors_url: str):
    contributors = get_json_from_url(contributors_url)
    if contributors == -1:
        return
    for contributor in contributors:
        id = contributor['id']
        user_url = contributor['url']
        if users.find_one({'_id': id}) is None:
            user = get_user(user_url)
            if user != -1:
                users.insert_one(user)

@check_rate_limit
def add_subscribers(subscribers_url: str):
    subscribers = get_json_from_url(subscribers_url)
    if subscribers == -1:
        return
    for subscriber in subscribers:
        id = subscriber['id']
        user_url = subscriber['url']
        if users.find_one({'_id': id}) is None:
            user = get_user(user_url)
            if user != -1:
                users.insert_one(user)

def add_repo_owner(user_url):
    user = get_user(user_url)
    if user == -1:
        return
    id = user['_id']
    if users.find_one({'_id': id}) is None:
        users.insert_one(user)
        logging.info('An user was added by add_repo_owner!')

def process_gzip_files():
    for gz_json_file in gz_json_files:
        logging.debug('Processing: %s' % gz_json_file)

        file_path = join(cfg.data_path, gz_json_file)

        file_content = None
        json_array = None

        with gzip.open(file_path, 'r') as gz_file:
            file_content = gz_file.read().decode('utf-8')
            json_array = file_content.split('\n')

        json_array_length = len(json_array)

        # each element of the json_array is an event object which contains the interaction between an user and a project.
        for j in json_array:
            logging.debug('Total number of records: %s, currently processing: %s, progress: %s' % (json_array_length, j))

            json_data = json.loads(j)
            event_type = json_data['type']
            created_at = datetime.datetime.strptime(json_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            
            # add repository
            repo = json_data['repo']
            repo_url = repo['url']
            repo_dict = get_repo(repo_url, True)
            if repo_dict != -1:
                repo_id = repo_dict['_id']
                exist_repo = repositories.find_one({'_id': repo_id})
                if exist_repo is None:
                    repositories.insert_one(repo_dict)
                    logging.info('A repository was added!')

            # add user
            user = json_data['actor']
            user_url = user['url']
            user_dict = get_user(user_url, True)
            if user_dict != -1:
                user_id = user_dict['_id']
                exist_user = users.find_one({'_id': user_id})
                if exist_user is None:
                    user_dict['interactions'] = [{'type': event_type, 'repo': repo_id, 'created_at': created_at}]
                    users.insert_one(user_dict)
                    logging.info('An user was added!')
                else:
                    interactions = exist_user['interactions'] if 'interactions' in exist_user else []
                    interactions.append({'type': event_type, 'repo': repo_id, 'created_at': created_at})
                    lookup_filter = {'_id': user_id}
                    new_values = {'$set': {'interactions': interactions}}
                    users.update_one(lookup_filter, new_values)
                    logging.info('An user was updated!')

if __name__ == "__main__":
    process_gzip_files()