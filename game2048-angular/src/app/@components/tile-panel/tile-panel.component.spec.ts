import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TilePanelComponent } from './tile-panel.component';

describe('TilePanelComponent', () => {
    let component: TilePanelComponent;
    let fixture: ComponentFixture<TilePanelComponent>;

    beforeEach(async(() => {
        TestBed.configureTestingModule({
            declarations: [TilePanelComponent]
        })
            .compileComponents();
    }));

    beforeEach(() => {
        fixture = TestBed.createComponent(TilePanelComponent);
        component = fixture.componentInstance;
        fixture.detectChanges();
    });

    it('should create', () => {
        expect(component).toBeTruthy();
    });
});
