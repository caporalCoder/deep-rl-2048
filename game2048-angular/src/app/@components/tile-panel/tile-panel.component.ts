import { ChangeDetectionStrategy, Component, Input, OnInit } from '@angular/core';
import { Tile } from '../../@stores/tile.model';
import { SIZE } from '../../@services/game.service';
import { animate, keyframes, style, transition, trigger } from '@angular/animations';

@Component({
    selector: 'dx-tile-panel',
    templateUrl: './tile-panel.component.html',
    styleUrls: ['./tile-panel.component.scss'],
    changeDetection: ChangeDetectionStrategy.OnPush,
    preserveWhitespaces: false,
    animations: [
        trigger('tileAnimate', [
            transition('unmerged => merged', [
                animate('200ms', keyframes([
                    style({transform: 'scale(1)', offset: 0}),
                    style({transform: 'scale(1.4)', offset: 0.5}),
                    style({transform: 'scale(1)', offset: 1.0})
                ]))
            ]),
            transition('void => *', [
                style({opacity: 0, transform: 'scale(0)'}),
                animate(300, style({opacity: 1, transform: 'scale(1)'}))
            ])
        ])
    ]
})


export class TilePanelComponent implements OnInit {

    @Input() tiles: Tile[];

    @Input() tileSize: number;

    constructor() {
    }

    ngOnInit() {
    }

    public getTileTransformStyle( tile: Tile ): string {
        const x = (tile.position % SIZE) * this.tileSize + 'px';
        const y = Math.floor(tile.position / SIZE) * this.tileSize + 'px';
        return `translate(${x}, ${y})`;
    }

    public trackByFn( index: number, tile: Tile ): string {
        return tile.id;
    }
}
