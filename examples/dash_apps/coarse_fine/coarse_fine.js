window.dash_clientside = Object.assign({}, window.dash_clientside, {
	clientside: {
		// TODO -> fix doubble callback -> maybe check whether the range of the selected that is the same 
		// range of the figure?
		update_main_with_coarse: function (selectedData, mainFigID, coarseFigID ) {
			//define helper function (imports not allowed?)
			function getGraphDiv(gdID) {
				// see this link for more information https://stackoverflow.com/a/34002028 
				let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
				if (graphDiv.length > 1) {
					throw new SyntaxError("UpdateStore: multiple graphs with ID=" + gdID + " found; n=" + graphDiv.length + " (either multiple graphs with same ID's or current ID is a str-subset of other graph IDs)");
				} else if (graphDiv.length < 1) {
					throw new SyntaxError("UpdateStore: no graphs with ID=" + gdID + " found");
				}
				graphDiv = graphDiv?.[0]?.getElementsByClassName('js-plotly-plot')?.[0];
				const isDOMElement = el => el instanceof HTMLElement
				if (!isDOMElement) {
					throw new Error(`Invalid gdID '${gdID}'`);
				}
				return graphDiv;
			}
			//can be changed for something more sophisticated if needed, 
			//it does the job of determining if two arrays have the same values for now
			const compareArrays = (a, b) => {
				return JSON.stringify(a) === JSON.stringify(b);
			};


			if (selectedData.range) {
				console.warn("starting: coarse -> main");
				main_graphDiv = getGraphDiv(mainFigID);
				coarse_graphDiv= getGraphDiv(coarseFigID);

				let yrange = main_graphDiv.layout.yaxis.range;
				let xrange = main_graphDiv.layout.xaxis.range;

				let cyrange = coarse_graphDiv.layout.yaxis.range;
				let cxrange = coarse_graphDiv.layout.xaxis.range;


				let currentSelections = coarse_graphDiv.layout.selections;
				let sxrange = [];
				let syrange = [];
				
				// console.log(currentSelections);
				if(currentSelections){
					sxrange = [currentSelections[0].x0, currentSelections[0].x1].sort();
					syrange = [currentSelections[0].y0, currentSelections[0].y1].sort();
				}

				// console.log("selection box ",selectedData);
				// console.log("xrange ", xrange);
				// console.log("sxrange ", sxrange);
				// console.log("cxrange ", cxrange);
				// console.log("yrange ", yrange);
				// console.log("syrange", syrange);
				// console.log("cyrange", cyrange);
				
				// console.log("selected data:");
				// console.log(selectedData.range.x.toString());
				// console.log("layout:")
				// console.log(grapDiv.layout.xaxis.range.toString());
				// console.log(graphDiv.layout.xaxis.range[0]);
				// console.log(graphDiv.layout.xaxis.range[1]);
				// console.log(graphDiv.layout.xaxis.range.toString() === selectedData.range.x.toString());
				// console.log(graphDiv.layout);

				/* 2 conditions for an update of the main graph: 
					
					1. the selection range is not the same as the main graph's range (before the update).

					2. the selection range is not exactly the same as the coarse graph's range 
					(which should be the same as the original main graph's range). 
					A selection range like this will be linked to the reset of the axes, meaning it comes from the 
					"main -> coarse" (update_coarse_selectbox) callback.
				 */
				if ((!compareArrays(sxrange, xrange) || !compareArrays(syrange, yrange)) && 
					(!compareArrays(syrange, cyrange) || !compareArrays(sxrange, cxrange))){ 
					console.warn("coarse -> main");
					Plotly.relayout(main_graphDiv, {
						'xaxis.range[0]': sxrange[0], 'xaxis.range[1]': sxrange[1],
						'yaxis.range[0]': syrange[0], 'yaxis.range[1]': syrange[1],
					}
					);
				}
			}
			return mainFigID;
		},
		update_coarse_selectbox: function (mainFigRelayout, coarseFigID, mainFigID) {
			//define helper function (imports not allowed?)
			function getGraphDiv(gdID) {
				// see this link for more information https://stackoverflow.com/a/34002028 
				let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
				if (graphDiv.length > 1) {
					throw new SyntaxError("UpdateStore: multiple graphs with ID=" + gdID + " found; n=" + graphDiv.length + " (either multiple graphs with same ID's or current ID is a str-subset of other graph IDs)");
				} else if (graphDiv.length < 1) {
					throw new SyntaxError("UpdateStore: no graphs with ID=" + gdID + " found");
				}
				graphDiv = graphDiv?.[0]?.getElementsByClassName('js-plotly-plot')?.[0];
				const isDOMElement = el => el instanceof HTMLElement
				if (!isDOMElement) {
					throw new Error(`Invalid gdID '${gdID}'`);
				}
				return graphDiv;
			}
			//can be changed for something more sophisticated if needed, 
			//it does the job of determining if two arrays have the same values for now
			const compareArrays = (a, b) => {
				return JSON.stringify(a) === JSON.stringify(b);
			};
			// external_scripts=[{'src':'../utils/callback_utils','type':'module'}]

			
			console.warn('starting: main -> coarse');
			let coarse_graphDiv = getGraphDiv(coarseFigID);
			let main_graphDiv = getGraphDiv(mainFigID);
			let yrange = main_graphDiv.layout.yaxis.range;
			let xrange = main_graphDiv.layout.xaxis.range;
			let currentSelections = coarse_graphDiv.layout.selections;
			let sxrange = [];
			let syrange = [];
			
			// console.log(currentSelections);
			if(currentSelections){
				sxrange = [currentSelections[0].x0, currentSelections[0].x1].sort();
				syrange = [currentSelections[0].y0, currentSelections[0].y1].sort();
			}

			// check if reset axis was triggered
			if (mainFigRelayout['xaxis.autorange'] && mainFigRelayout['xaxis.showspikes'] == false) {
				console.log("reset axis");
				xrange = coarse_graphDiv.layout.xaxis.range;
				yrange = coarse_graphDiv.layout.yaxis.range;
			}
			console.log("relayout", mainFigRelayout);
			console.log("xrange", xrange);
			console.log("sxrange", sxrange);
			console.log("yrange", yrange);
			console.log("syrange", syrange);
			
			// console(external_scripts.compareArrays(sxrange, xrange));
			
			// if (mainFigRelayout['xaxis.autorange'] && mainFigRelayout['xaxis.showspikes'] == false) {
			// 	console.log("reset axis");
			// 	xrange = coarse_graphDiv.layout.xaxis.range;
			// 	yrange = coarse_graphDiv.layout.yaxis.range;
			// }
			if(!compareArrays(sxrange, xrange) || !compareArrays(syrange,yrange)){
				// Update the layout without triggering another relayout event
				console.warn('main -> coarse');
				// document.dispatchEvent();
				Plotly.relayout(
					coarse_graphDiv,
					{
						'selections': [
							{
								"xref": "x",
								"yref": "y",
								"line": {
									"width": 1,
									"dash": "dot"
								},
								"type": "rect",
								"x0": xrange[0],
								"x1": xrange[1],
								"y0": yrange[0],
								"y1": yrange[1]
							}
						]
					}
				);
			}
			
			return coarseFigID;
		},
		set_coarse_range: function (coarsefigure, mainFigID, coarseFigID ){
			function getGraphDiv(gdID) {
				// see this link for more information https://stackoverflow.com/a/34002028 
				let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
				if (graphDiv.length > 1) {
					throw new SyntaxError("UpdateStore: multiple graphs with ID=" + gdID + " found; n=" + graphDiv.length + " (either multiple graphs with same ID's or current ID is a str-subset of other graph IDs)");
				} else if (graphDiv.length < 1) {
					throw new SyntaxError("UpdateStore: no graphs with ID=" + gdID + " found");
				}
				graphDiv = graphDiv?.[0]?.getElementsByClassName('js-plotly-plot')?.[0];
				const isDOMElement = el => el instanceof HTMLElement
				if (!isDOMElement) {
					throw new Error(`Invalid gdID '${gdID}'`);
				}
				return graphDiv;
			}
			//can be changed for something more sophisticated if needed, 
			//it does the job of determining if two arrays have the same values for now
			const compareArrays = (a, b) => {
				return JSON.stringify(a) === JSON.stringify(b);
			};

			let coarse_graphDiv = getGraphDiv(coarseFigID);
			let main_graphDiv = getGraphDiv(mainFigID);
			let myrange = main_graphDiv.layout.yaxis.range;
			let mxrange = main_graphDiv.layout.xaxis.range;
			let cyrange = coarse_graphDiv.layout.yaxis.range;
			let cxrange = coarse_graphDiv.layout.xaxis.range;
			
			if(!compareArrays(mxrange, cxrange) || !compareArrays(myrange,cyrange)){
				console.log("updating coarse range");
				Plotly.relayout(coarse_graphDiv, {
					'xaxis.range[0]': mxrange[0], 'xaxis.range[1]': mxrange[1],
					'yaxis.range[0]': myrange[0], 'yaxis.range[1]': myrange[1],
				}
				);
				// let coarse_graphDiv2 = getGraphDiv(coarseFigID);
				// let cyrange2 = coarse_graphDiv2.layout.yaxis.range;
				// let cxrange2 = coarse_graphDiv2.layout.xaxis.range;
				// console.log("coarse range x: " + cxrange2);
				// console.log("coarse range y:" + cyrange2);
				// console.log("main range x: " + mxrange);
				// console.log("main range y: " + myrange);
			}
			return mainFigID;		
		}
	}
});