import { Component } from '@angular/core';
import { Rankings, Ranking } from './data-model';
import { SearchService } from './services/search-service';
import * as _ from 'underscore';

import { PagerService } from './services/pager-service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  position = 'below'
  title = 'One Engine to find them all';
  private searchQuery: string;
  private error: boolean = false;
  private errorMessage: string;
  constructor(private searchService: SearchService, private pagerService: PagerService) { }
  private allRankings: Ranking[];
  pager: any = {};
  pagedRankings: Ranking[];

  onSearch(query: String, alpha: number) {
    this.allRankings = [];
    this.pagedRankings = [];
    this.error = false;
    if (query != "") {
      let rankings = this.searchService.search(query, alpha).subscribe(rankings => { this.allRankings = rankings; this.setPage(1); },
        error => { this.errorMessage = <any>error; this.error = true; });
    }
  }
  setPage(page: number) {
        if (page < 1 || page > this.pager.totalPages) {
            return;
        }
        this.pager = this.pagerService.getPager(this.allRankings.length, page);
        this.pagedRankings = this.allRankings.slice(this.pager.startIndex, this.pager.endIndex + 1);
    }
}
