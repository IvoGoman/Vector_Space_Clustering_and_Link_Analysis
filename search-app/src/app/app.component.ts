import { Component } from '@angular/core';
import {Rankings, Ranking} from './data-model';
import { SearchService } from './search-service';

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
  private rankings: Ranking[];
  private errorMessage: string;
  constructor(private service: SearchService) {}

  onSearch(query: String) {
    console.log(query);
    let rankings = this.service.search(query).subscribe(rankings=>{this.rankings = rankings; this.error=false;},
    error=> {this.errorMessage = <any> error; this.error = true; this.rankings = []});
    console.log(rankings)
  }
}
