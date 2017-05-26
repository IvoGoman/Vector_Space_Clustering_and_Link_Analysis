import { Injectable } from '@angular/core';
import { Http, Response, Headers, RequestOptions } from '@angular/http';
import { Observable } from 'rxjs/Observable';
import 'rxjs/add/operator/map';
import { Ranking, Rankings } from '../data-model';

// source: https://github.com/cornflourblue/angular2-pagination-example

@Injectable()
export class SearchService {
    private searchEndpoint = "http://127.0.0.1:8080/query"

    constructor(private http: Http) { }

    search(query: String, alpha: Number): Observable<Ranking[]> {
        let header = new Headers({ 'Content-Type': 'application/json' });
        let options = new RequestOptions({ headers: header });
        let json = { "query": query, "alpha": alpha }
        return this.http.post(this.searchEndpoint, json, options).map(res => this.extractRankings(res) ) .catch(this.handleErrorObservable)
    }

    private extractRankings(res: Response): Ranking[] {
        let rankings =  res.json().results.map(r => this.extractRanking(r));
        return rankings
    }

    private extractRanking(r:any): Ranking {
        let ranking = new Ranking(
        r.id, r.text,
            r.score,
             r.pagerank,
             r.cosine
        );
        return ranking;
    }

    private handleErrorObservable(error: Response | any) {
        console.error(error.message || error);
        return Observable.throw(error.message || error);
    }
}