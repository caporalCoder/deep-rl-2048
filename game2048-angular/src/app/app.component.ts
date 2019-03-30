import { Component, HostBinding, OnInit } from '@angular/core';
import { GameService } from './@services/game.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  @HostBinding('class.game-2048')
    get game2048Class(): boolean {
        return true;
    }

    constructor( private gameService: GameService ) {
    }

    public ngOnInit() {
      this.newGame();
    }

    public newGame(): void {
        this.gameService.newGame();
    }
}
