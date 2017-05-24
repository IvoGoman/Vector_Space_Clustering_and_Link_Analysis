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
  title = 'One Engine to find the answers';
  private searchQuery: string;
  private rankings: Ranking[];
  private errorMessage: string;
  constructor(private service: SearchService) {}

  onSearch(query: String) {
    console.log(query);
    let rankings = this.service.search(query).subscribe(rankings=>this.rankings = rankings,
    error=> this.errorMessage = <any> error);
    console.log(rankings)
  }
}
