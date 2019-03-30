import { ChangeDetectionStrategy, Component, Input, OnInit } from '@angular/core';
import { SIZE } from '../../@services/game.service';

@Component({
    selector: 'dx-cell-panel',
    templateUrl: './cell-panel.component.html',
    styleUrls: ['./cell-panel.component.scss'],
    changeDetection: ChangeDetectionStrategy.OnPush,
    preserveWhitespaces: false,
})
export class CellPanelComponent implements OnInit {

    @Input() cells: string[];

    @Input() size: number;

    constructor() {
    }

    ngOnInit() {
    }

    public getCellTransformStyle( index: number ): string {
        const x = (index % SIZE) * this.size + 'px';
        const y = Math.floor(index / SIZE) * this.size + 'px';
        return `translate(${x}, ${y})`;
    }
}
