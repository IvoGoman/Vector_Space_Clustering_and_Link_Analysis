export class Rankings {
    rankings: Array<Ranking> ;
    constructor(){}
}

export class Ranking{
    pr_score: number;
    cos_score: number;
    score: number;
    text: string;
    id: string;
    constructor(id:string, text:string, score:number, cos_score:number, pr_score:number){
        this.id = id;
        this.text = text;
        this.score = score;
        this.pr_score = pr_score;
        this.cos_score = cos_score;

    }
}