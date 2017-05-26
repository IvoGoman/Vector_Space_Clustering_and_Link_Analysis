import json
import os
from flask import Flask, request, make_response, jsonify
from search_modules import Query
from search_engine import SearchEngine

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def query():
    """ Receives Query from the Frontend"""
    req = request.get_json(silent=True, force=True)
    query_string = Query(req.get('query'))
    result = SEARCH_ENGINE.run_query(query_string,alpha_pr=0.2)
    # result = jsonify(result.to_dict())
    result = create_response(result)
    result = json.dumps(result, indent=4)
    response = make_response(result)
    response.headers['Content-Type'] = 'application/json'
    return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

def create_response(result):
    """Create the response Json for a given ranking"""
    rankings = []
    for row in result.iterrows():
        ranking = {"id": str(row[0]), "text": row[1]
                   ['text'], "score": row[1]['score']}
        rankings.append(ranking)
    return {
        "results": rankings
    }


if __name__ == '__main__':
    print('Starting Search Engine')
    SEARCH_ENGINE = SearchEngine()
    PORT = 8080
    print('Search Engine ready for Queries')
    app.run(debug=False, port=PORT, host='127.0.0.1')
