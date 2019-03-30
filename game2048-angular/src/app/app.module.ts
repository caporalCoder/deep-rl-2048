import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { StoreModule } from '@ngrx/store';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

import { BoardComponent } from './@components/board/board.component';
import { CellPanelComponent } from './@components/cell-panel/cell-panel.component';
import { TilePanelComponent } from './@components/tile-panel/tile-panel.component';

import { GameService } from './@services/game.service';
import { reducers } from './@stores';
import { HeaderComponent } from './@components/header/header.component';
import { OverPanelComponent } from './@components/over-panel/over-panel.component';


@NgModule({
    declarations: [
        AppComponent,
        BoardComponent,
        CellPanelComponent,
        TilePanelComponent,
        HeaderComponent,
        OverPanelComponent
    ],
    imports: [
        BrowserModule,
        AppRoutingModule,
        BrowserAnimationsModule,
        StoreModule.forRoot(reducers)
    ],
    providers: [
        GameService
    ],
    bootstrap: [AppComponent]
})
export class AppModule { }
