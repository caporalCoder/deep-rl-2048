/**
 * author: Foromo Daniel Soromou
 */

import {
    ChangeDetectionStrategy,
    Component, 
    EventEmitter, 
    HostBinding, 
    OnInit, 
    Output
} from '@angular/core';

import { ActivatedRoute, Router } from '@angular/router';

@Component({
    selector: 'dx-header',
    templateUrl: './header.component.html',
    styleUrls: ['./header.component.scss'],
    changeDetection: ChangeDetectionStrategy.OnPush,
    preserveWhitespaces: false,
})

export class HeaderComponent implements OnInit {

    @Output() public startNewGame = new EventEmitter<any>();

    @HostBinding('class.game-2048-header')
    get game2048HeaderClass(): boolean {
        return true;
    }

    constructor( private router: Router,
                 private route: ActivatedRoute ) {
    }

    public ngOnInit() {
    }

    public newGame(): void {
        if (this.router.url === '/games/2048') {
            this.startNewGame.emit(true);
        } else {
            this.router.navigate(['./'], {relativeTo: this.route});
        }
    }

    public checkAbout() {
        this.router.navigate(['./about'], {relativeTo: this.route});
    }
}
