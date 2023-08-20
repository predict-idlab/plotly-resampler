window.dash_clientside = Object.assign({}, window.dash_clientside, {
	clientside: {
		// TODO -> fix doubble callback -> maybe check whether the range of the selected that is the same 
		// range of the figure?
		coarse_to_main: function (selectedData, mainFigID, coarseFigID, linkedIndices) {
			/*
				------------ HELPERS -----------
			*/
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

			const roundTo5Decimals = (a) => {
				if (isNumber(a)) {
					return +a.toFixed(5);
				} else {
					return a;
				}

			};

			const isNumber = (currentValue) => typeof (currentValue) == "number";

			const isDate = (currentValue) => !isNan(Date.parse(currentValue));

			/**
			 * 
			 * @param {Array} array : array to be sorted 
			 * @param {Array} reference : array to use as reference for sorting direction
			 * @returns 
			 */
			const timeSeriesRangeSort = (array, reference) => {
				if (array.every(isNumber || isDate)) {
					array = array.sort((a, b) => {
						if (isNumber(a) && !isNaN(a) && isNumber(b) && !isNaN(b)) {
							if (reference.every(isNumber)) {
								if (reference[0] > reference[1]) {
									return b - a;
								}
							}
							return a - b;

						} else if (isDate(a) && isDate(b)) {
							if (reference.every(isDate)) {
								if (Date.parse(reference[0]) > Date.parse(reference[1])) {
									return Date.parse(b) - Date.parse(a);
								}
							}
							return Date.parse(a) - Date.parse(b);
						}
					});
				}
				return array;
			}

			/**
			 * 
			 * @param {object} layout : layout object from which the x and y ranges should be extracted
			 * @param {Array} indexList : list of subplot indices of which the x and y range should be extracted. undefined = all subplots
			 * @returns an array containing an array of xranges and an array of yranges of the figure the layout comes from
			 */
			const getFigureInfo = (layout, indexList = undefined) => {
				const yrange = [];
				const xrange = [];

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						const axisNumber = key.slice(5, 6) ? key.slice(5, 6) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							yrange.push({ [axisNumber]: axisRange });
						}
					} else if (key.startsWith("xaxis")) {
						const axisNumber = key.slice(5, 9) ? key.slice(5, 9) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							xrange.push({ [axisNumber]: axisRange });
						}
					}
				});
				return [xrange, yrange];
			};
			/*
				------------ CALLBACK -----------
			*/
			//obtain the graph components from the DOM

			//for future: merge callbacks and check trigger prop!
			// console.log(window.dash_clientside.callback_context.triggered[0].prop_id);
			const main_graphDiv = getGraphDiv(mainFigID);
			const coarse_graphDiv = getGraphDiv(coarseFigID);
			let updateCondition = false;
			let updates = {};
			if (selectedData) {
				if (selectedData.range) {
					// console.warn("starting: coarse -> main");

					//obtain the triggers from selectedData. should only be 1 in most cases, but we still create a list
					let triggerCols = [...new Set(Object.keys(selectedData.range).map(item => +(item.substring(1) || 1) - 1))];

					const updateData = [];
					linkedIndices.forEach((item, i) => {
						updateData.push({
							"columnIndex": +i,
							"linkIndex": +item,
							"mainSubplotIndex": +linkedIndices.length * item + i
						})
					})

					let filteredTriggerData = updateData.filter(obj => triggerCols.includes(obj.columnIndex));

					const filteredMainTriggers = filteredTriggerData.map(obj => obj.mainSubplotIndex);

					let [xrange, yrange] = getFigureInfo(main_graphDiv.layout, filteredMainTriggers);
					filteredTriggerData = filteredTriggerData.map(obj => {
						const subplotIndex = obj.mainSubplotIndex;
						// const columnIndex = obj.columnIndex;
						const yrangeval = yrange.find(o => o.hasOwnProperty(subplotIndex));
						const xrangeval = xrange.find(o => o.hasOwnProperty(subplotIndex));

						if (xrangeval) {
							obj["mxrange"] = xrangeval[subplotIndex].map(roundTo5Decimals);
							obj["myrange"] = yrangeval[subplotIndex].map(roundTo5Decimals);
							obj["sxrange"] = [];
							obj["syrange"] = [];
						}
						return obj
					});

					//get the selections currently on the overviews => fill in the filteredTriggerData where needed
					let currentSelections = coarse_graphDiv.layout.selections;
					if (currentSelections) {

						filteredTriggerData = filteredTriggerData.map(obj => {
							const subplotIndex = obj.columnIndex + 1;
							const filteredSelection = currentSelections.filter(selection => {
								const sxrange = timeSeriesRangeSort([selection.x0, selection.x1], obj.mxrange).map(roundTo5Decimals);
								const syrange = timeSeriesRangeSort([selection.y0, selection.y1], obj.myrange).map(roundTo5Decimals);

								//Plotly accepts both x and x1 as indices, 
								//so if we are looking for selections the first axis, check both options
								//if no subplots (only 1 figure), no xref!! => undefined selection.xref valid if columnIndex == 0
								if (obj.columnIndex == 0) {
									return (selection.xref === 'x' || selection.xref === `x${subplotIndex}` || !selection.xref)
										&& (!compareArrays(sxrange, obj.mxrange) || !compareArrays(syrange, obj.myrange));
								} else {
									return selection.xref === `x${subplotIndex}`
										&& (!compareArrays(sxrange, obj.mxrange) || !compareArrays(syrange, obj.myrange));
								}

							});

							if (filteredSelection.length > 0) {
								obj["sxrange"] = timeSeriesRangeSort([filteredSelection[0].x0, filteredSelection[0].x1], obj.mxrange).map(roundTo5Decimals);
								obj["syrange"] = timeSeriesRangeSort([filteredSelection[0].y0, filteredSelection[0].y1], obj.myrange).map(roundTo5Decimals);

							}
							return obj;
						});
					}


					/**
					 * 1. check which graph the selection change comes from
					 */


					/* 2 conditions for an update of the main graph: 
						
						1. the selection range is not the same as the main graph's range (before the update).

						2. the selection range is not exactly the same as the coarse graph's range 
						(which should be the same as the original main graph's range). 
						A selection range like this will be linked to the reset of the axes, meaning it comes from the 
						"main -> coarse" (update_coarse_selectbox) callback.

						DIFFERENT RELAYOUT MESSAGES: if the selection xrange is the same as the main graph xrange, 
						but this is not the case in the y direction, relayout should only contain the y range change.
						This is done to prevent an unnecessary aggregation in the Plotly resampler back-end
					*/


					filteredTriggerData.forEach((obj, i) => {
						const subplotIndex = +obj.mainSubplotIndex === 0 ? "" : String(obj.mainSubplotIndex + 1);

						let xaxisKey0 = `xaxis${subplotIndex}.range[0]`;
						let xaxisKey1 = `xaxis${subplotIndex}.range[1]`;
						let yaxisKey0 = `yaxis${subplotIndex}.range[0]`;
						let yaxisKey1 = `yaxis${subplotIndex}.range[1]`;

						if ((!compareArrays(obj.sxrange, obj.mxrange) && !compareArrays(obj.syrange, obj.myrange))) {
							updateCondition = true;
							updates = {
								...updates,
								[xaxisKey0]: obj.sxrange[0],
								[xaxisKey1]: obj.sxrange[1],
								[yaxisKey0]: obj.syrange[0],
								[yaxisKey1]: obj.syrange[1],
							};
						} else if (!compareArrays(obj.sxrange, obj.mxrange)) {
							updateCondition = true
							updates = {
								...updates,
								[xaxisKey0]: obj.sxrange[0],
								[xaxisKey1]: obj.sxrange[1],
							};
						} else if (!compareArrays(obj.syrange, obj.myrange)) {
							updateCondition = true
							updates = {
								...updates,
								[yaxisKey0]: obj.syrange[0],
								[yaxisKey1]: obj.syrange[1],
							};
						}

					});
				}

			}
			if (updateCondition) {
				// Update the layout without triggering another relayout event
				// console.warn('coarse -> main');
				// document.dispatchEvent(); doesnt work
				Plotly.relayout(
					main_graphDiv, updates
				);
			}

			return mainFigID;
		},
		main_to_coarse: function (mainFigRelayout, coarseFigID, mainFigID, linkedIndices) {
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

			const roundTo5Decimals = (a) => {
				if (isNumber(a)) {
					return +a.toFixed(5);
				} else {
					return a;
				}

			};

			const isNumber = (currentValue) => typeof (currentValue) == "number";
			const isDate = (currentValue) => !isNan(Date.parse(currentValue));
			const timeSeriesRangeSort = (array, reference) => {
				if (array.every(isNumber || isDate)) {
					array = array.sort((a, b) => {
						if (typeof (a) == "number" && !isNaN(a) && typeof (b) == "number" && !isNaN(b)) {
							// console.log("sorting by number value");
							if (reference.every(isNumber)) {
								if (reference[0] > reference[1]) {
									return b - a;
								}
							}
							return a - b;

						} else if (!isNaN(Date.parse(a)) && !isNan(Date.parse(b))) {
							// console.log("sorting by parsed date");
							if (reference.every(isDate)) {
								if (Date.parse(reference[0]) > Date.parse(reference[1])) {
									return Date.parse(b) - Date.parse(a);
								}
							}
							return Date.parse(a) - Date.parse(b);
						}
					});
				}
				return array;
			}

			/**
			 * 
			 * @param {object} layout : layout object from which the x and y ranges should be extracted
			 * @param {Array} indexList : list of subplot indices of which the x and y range should be extracted. undefined = all subplots
			 * @returns an array containing an array of xranges and an array of yranges of the figure the layout comes from
			 */
			const getFigureInfo = (layout, indexList = undefined) => {
				const yrange = [];
				const xrange = [];

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						const axisNumber = key.slice(5, 6) ? key.slice(5, 6) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							yrange.push({ [axisNumber]: axisRange });
						}
					} else if (key.startsWith("xaxis")) {
						const axisNumber = key.slice(5, 9) ? key.slice(5, 9) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							xrange.push({ [axisNumber]: axisRange });
						}
					}
				});
				return [xrange, yrange];
			};

			/**
			 * (unused but may be useful in the future?) 
			 * @param {object} layout: object containing the layout information of a figure
			 * @returns an array containing the number of rows and columns in the given figure
			 */
			const getGridLayout = (layout) => {
				const xaxes = new Set();
				const yaxes = new Set();
				//TODO: adapt for when no xaxis in layout (no subplots)
				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						yaxes.add(JSON.stringify(layout[key].domain));
					} else if (key.startsWith("xaxis")) {
						xaxes.add(JSON.stringify(layout[key].domain));
					}
				});
				return [yaxes.size, xaxes.size]
			}

			//function to extract the subplot index from which the relayout comes from
			const getTriggerSubplot = (relayout) => {
				let triggerPlot = new Set();
				const patternRange = /(y|x)?axis|\.range\[\d\]/g;
				const patternAutorange = /(y|x)?axis|\.autorange/g;
				Object.keys(relayout).forEach(key => {
					if (key.startsWith("yaxis") || key.startsWith("xaxis")) {
						let axisNumber;
						if (key.includes(".range")) {
							axisNumber = key.replace(patternRange, '');
							axisNumber = axisNumber ? +axisNumber - 1 : 0;
							triggerPlot.add(axisNumber);
						} else if (key.includes(".autorange")) {
							axisNumber = key.replace(patternAutorange, '');
							axisNumber = axisNumber ? +axisNumber - 1 : 0;
							triggerPlot.add(axisNumber);
						}
					}
				});
				return Array.from(triggerPlot);
			}
			//tried to move helpers to external file... failed
			// external_scripts=[{'src':'../utils/callback_utils','type':'module'}]

			/*
				------------ CALLBACK -----------
			*/
			// console.warn('starting: main -> coarse');

			let coarse_graphDiv = getGraphDiv(coarseFigID);
			let main_graphDiv = getGraphDiv(mainFigID);

			let trigger = getTriggerSubplot(mainFigRelayout);

			// linkedIndices is an array showing which subplots of the main graph are linked with a coarse view
			// obtained from a Store in the client
			//list of objects compiling all data needed to create an update for relayout
			const updateData = [];
			linkedIndices.forEach((item, i) => {
				updateData.push({
					"columnIndex": +i,
					"linkIndex": +item,
					"mainSubplotIndex": +linkedIndices.length * item + i
				})
			});

			//filter the trigger list to only the ones that have a coarse view linked to them
			let filteredTriggerData = updateData.filter(obj => trigger.includes(obj.mainSubplotIndex));

			//cross-filtered list of triggers (subplot indices that triggered a relayout AND are linked to an overview)
			const filteredTriggers = filteredTriggerData.map(obj => obj.mainSubplotIndex);

			//obtain the x & yrange of the triggered and linked subplots
			let [xrange, yrange] = getFigureInfo(main_graphDiv.layout, filteredTriggers);
			let [cxrange, cyrange] = getFigureInfo(coarse_graphDiv.layout);

			filteredTriggerData = filteredTriggerData.map(obj => {
				const subplotIndex = obj.mainSubplotIndex;
				const columnIndex = obj.columnIndex;
				const yrangeval = yrange.find(o => o.hasOwnProperty(subplotIndex));
				const xrangeval = xrange.find(o => o.hasOwnProperty(subplotIndex));
				const cyrangeval = cyrange.find(o => o.hasOwnProperty(columnIndex));
				const cxrangeval = cxrange.find(o => o.hasOwnProperty(columnIndex));


				if (xrangeval) {
					obj["mxrange"] = xrangeval[subplotIndex].map(roundTo5Decimals);
					obj["myrange"] = yrangeval[subplotIndex].map(roundTo5Decimals);
					obj["cxrange"] = cxrangeval[columnIndex].map(roundTo5Decimals);
					obj["cyrange"] = cyrangeval[columnIndex].map(roundTo5Decimals);
					obj["sxrange"] = [];
					obj["syrange"] = [];
				}
				return obj
			});

			// obtain the selections that should be changed (using the subplot index mapped in filteredTriggeredData)
			// used to check if it really is necessary to change the selectionbox!
			let currentSelections = coarse_graphDiv.layout.selections;
			let noSubplots = false;
			if (currentSelections) {

				filteredTriggerData = filteredTriggerData.map(obj => {
					const subplotIndex = obj.columnIndex;
					const filteredSelection = currentSelections.filter(selection => {
						if (subplotIndex === 0) {
							return selection.xref === `x${subplotIndex + 1}` || selection.xref === "x" || !selection.xref
						} else {
							return selection.xref === `x${subplotIndex + 1}`
						}

					});

					if (filteredSelection.length > 0) {
						//sort the selection range according to the direction the main graph is in (overview range could also be used as reference)
						obj["sxrange"] = timeSeriesRangeSort([filteredSelection[0].x0, filteredSelection[0].x1], obj.mxrange).map(roundTo5Decimals);
						obj["syrange"] = timeSeriesRangeSort([filteredSelection[0].y0, filteredSelection[0].y1], obj.myrange).map(roundTo5Decimals);
						if (!filteredSelection.xref) {
							noSubplots = true;
						}
					}
					return obj
				});
			}

			// check if autoscale/axis reset was triggered 
			// autorange is triggered per axis,
			// but showspikes only appears in relayout for xaxis and yaxis
			let notAutorange = true;
			let notShowspikes = true;
			filteredTriggerData.forEach(obj => {
				const subplotIndex = +obj.mainSubplotIndex === 0 ? "" : String(obj.mainSubplotIndex + 1);
				if (mainFigRelayout[`xaxis${subplotIndex}.autorange`]
					&& mainFigRelayout[`yaxis${subplotIndex}.autorange`]) {
					//use the coarse range as a reference for 
					//the range to be used in the relayout for the selection box.
					//not accurate to the range shown in the main graph, but it's a way to
					//make the selectionbox easier to see/grab
					obj.myrange = obj.cyrange;
					notAutorange = false;
					if (mainFigRelayout[`xaxis${subplotIndex}.showspikes`] === false
						&& mainFigRelayout[`yaxis${subplotIndex}.showspikes`] === false) {
						//simply resets the box to the full range again
						obj.mxrange = obj.cxrange;
						notShowspikes = false;
					}
				}
			});

			//i should put the full original currentSelections in the update, 
			//THEN only change the trigger ones?
			let updateCondition = false;
			update = { 'selections': currentSelections || [] }
			filteredTriggerData.forEach((obj, i) => {
				// console.log(!compareArrays(obj.sxrange,obj.mxrange))
				// console.log(!compareArrays(obj.syrange, obj.myrange));
				// console.log(notAutorange == notShowspikes);
				if ((!compareArrays(obj.sxrange, obj.mxrange) && notAutorange == notShowspikes) || !compareArrays(obj.syrange, obj.myrange)) {
					updateCondition = true;
					if (noSubplots) {
						update['selections'][0] =
						{
							"line": {
								"width": 1,
								"dash": "dot"
							},
							"type": "rect",
							"x0": obj.mxrange[0],
							"x1": obj.mxrange[1],
							"y0": obj.myrange[0],
							"y1": obj.myrange[1]
						};
					} else {
						const columnIndex = +obj.columnIndex === 0 ? "" : String(obj.columnIndex + 1);
						const selectionIndex = update['selections'].findIndex(obj => obj.xref === `x${columnIndex}`);

						update['selections'][selectionIndex] =
						{
							"xref": `x${columnIndex}`,
							"yref": `y${columnIndex}`,
							"line": {
								"width": 1,
								"dash": "dot"
							},
							"type": "rect",
							"x0": obj.mxrange[0],
							"x1": obj.mxrange[1],
							"y0": obj.myrange[0],
							"y1": obj.myrange[1]
						};
					}
				}

			});
			// console.log(update);

			if (updateCondition) {
				// Update the layout without triggering another relayout event
				// console.warn('main -> coarse');
				// document.dispatchEvent(); doesnt work
				Plotly.relayout(
					coarse_graphDiv, update
				);
			}

			return coarseFigID;
		}
	}
});