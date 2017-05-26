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

  onSearch(query: String, alpha: number) {
    this.rankings = [];
    this.error = false;
    if(query != ""){
    let rankings = this.service.search(query, alpha).subscribe(rankings=>{this.rankings = rankings},
    error=> {this.errorMessage = <any> error; this.error = true;});
    }
  }
}
