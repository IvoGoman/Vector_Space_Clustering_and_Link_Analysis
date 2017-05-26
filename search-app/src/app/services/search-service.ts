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
        return this.http.post(this.searchEndpoint, json, options).map(res=> res.json().results).catch(this.handleErrorObservable)
    }

    private extractRankings(res: Response): Rankings {
        console.log(res.json().results)
        let rankings_ =  res.json().results.map(this.extractRanking);
        console.log(rankings_);
        return <Rankings>({rankings: rankings_})
    }

    private extractRanking(r:any): Ranking {
        let ranking = <Ranking>({
            id:r.id,
            text: r.text,
            score: r.score
        });
        console.log(ranking);
        return ranking;
    }

    private handleErrorObservable(error: Response | any) {
        console.error(error.message || error);
        return Observable.throw(error.message || error);
    }
}