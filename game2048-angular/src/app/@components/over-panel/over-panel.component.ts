import { ChangeDetectionStrategy, Component, EventEmitter, HostBinding, OnInit, Output } from '@angular/core';
import { animate, state, style, transition, trigger } from '@angular/animations';

@Component({
    selector: 'dx-over-panel',
    templateUrl: './over-panel.component.html',
    styleUrls: ['./over-panel.component.scss'],
    changeDetection: ChangeDetectionStrategy.OnPush,
    preserveWhitespaces: false,
    animations: [
        trigger('showPanel', [
            state('in', style({opacity: 1})),
            transition('void => in', [
                style({opacity: 0}),
                animate('300ms 700ms ease-in')
            ])
        ])
    ]
})
export class OverPanelComponent implements OnInit {

    @Output() restart = new EventEmitter<any>();

    @HostBinding('@showPanel')
    get overPanelAnimation(): string {
        return 'in';
    }

    constructor() {
    }

    ngOnInit() {
    }

}
