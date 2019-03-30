/**
 * board.component
 */

import {
    AfterContentInit, ChangeDetectionStrategy, Component, ElementRef, HostBinding, OnDestroy, OnInit,
    ViewChild
} from '@angular/core';
import { GameService, SIZE } from '../../@services/game.service';
import { select, Store } from '@ngrx/store';
import * as from2048 from '../../@stores';
import { Observable } from 'rxjs';
import { Tile } from '../../@stores/tile.model';
import { DOWN_ARROW, LEFT_ARROW, RIGHT_ARROW, UP_ARROW } from '@angular/cdk/keycodes';
import { GameStats } from '../../@stores/game-stats.model';

@Component({
    selector: 'dx-board',
    templateUrl: './board.component.html',
    styleUrls: ['./board.component.scss'],
    changeDetection: ChangeDetectionStrategy.OnPush,
    preserveWhitespaces: false,
})

export class BoardComponent implements OnInit, AfterContentInit, OnDestroy {

    public tiles$: Observable<Tile[]>;

    public stats$: Observable<GameStats>;

    public cellSize: number;

    @ViewChild('wrapper') panelWrapperElmRef: ElementRef;

    @HostBinding('class.game-2048-board')
    get game2048BoardClass(): boolean {
        return true;
    }

    get cells(): string[] {
        return this.gameService.cells;
    }

    constructor( private gameService: GameService,
                 private store: Store<from2048.State> ) {
    }

    public ngOnInit() {
        this.tiles$ = this.store.pipe(select(from2048.getTiles));
        this.stats$ = this.store.pipe(select(from2048.getGameStats));

        this.newGame();
    }

    public ngAfterContentInit(): void {
        this.cellSize = (this.panelWrapperElmRef.nativeElement.offsetWidth - 24) / SIZE;
    }

    public ngOnDestroy(): void {
    }

    public handleKeydownOnWrapper( event: KeyboardEvent ): void {
        const keycode = event.keyCode;

        switch (keycode) {

            case LEFT_ARROW:
            case UP_ARROW:
            case RIGHT_ARROW:
            case DOWN_ARROW:
                this.gameService.move(keycode);
                event.preventDefault();
                return;

            default:
                return;
        }
    }

    public newGame(): void {
        this.gameService.newGame();
    }
}
