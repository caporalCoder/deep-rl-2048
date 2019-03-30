/**
 * author: Foromo Daniel Soromou
 */

import { Action } from '@ngrx/store';
import { Tile } from './tile.model';
import { GameStats } from './game-stats.model';

export enum GameActionTypes {
    ResetGame = '[2048] Result Game',
    SetGameStats = '[2048] Set Game Stat',
    AddTile = '[2048] Add Tile',
    UpdateTiles = '[2048] Update Tiles',
    ResetTilesStatus = '[2048] Rest Tiles Status'
}

export class ResetGame implements Action {
    readonly type = GameActionTypes.ResetGame;
}

export class SetGameStats implements Action {
    readonly type = GameActionTypes.SetGameStats;

    constructor(public payload: GameStats) {

    }
}

export class AddTile implements Action {
    readonly type = GameActionTypes.AddTile;

    constructor(public payload: Tile) {

    }
}

export class UpdateTiles implements Action {
    readonly type = GameActionTypes.UpdateTiles;

    constructor(public payload: { newPosition: number, oldTile: Tile, nextTile: Tile }) {

    }
}

export class ResetTileStatus implements Action {
    readonly type = GameActionTypes.ResetTilesStatus
}

export type GameActions = 
    ResetGame |
    SetGameStats |
    AddTile |
    UpdateTiles |
    ResetTileStatus