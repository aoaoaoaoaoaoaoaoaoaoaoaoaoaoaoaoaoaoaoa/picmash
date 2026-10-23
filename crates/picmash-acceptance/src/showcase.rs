//! Picmash's showcase: the comparison loop performed over the cat specimen.
//!
//! Comparison pairs the least-judged images first, so once every resident cat
//! has dueled, the Persian and the Sphynx admitted by a rescan, never judged,
//! must meet each other: that duel is the frontispiece.

use egui_demo::Showcase;
use egui_tester::{
    AppCommand, ArtifactStem, Condition, Graphics, Network, ReactionBudget, Result, Story, Testbed,
    WindowQuery, demand,
};
use picmash_contract::{Side, Target, UI_FINGERPRINT};
use std::{path::Path, time::Duration};

use crate::{
    Observation, STARTUP, TITLE, WITNESS,
    specimen::{Photograph, Specimen},
};

const PERSIAN: &str = "persian-amber-gaze.jpg";
const SPHYNX: &str = "sphynx-lavender.jpg";
const NEWCOMERS: [&str; 2] = [PERSIAN, SPHYNX];
const COLLECTION: &str = "cats";

type Duel<'app, 'bed> = Story<'app, 'bed, Observation, Showcase>;

pub fn perform(
    testbed: &Testbed,
    binary: &Path,
    specimen: &Specimen,
    showcase: Showcase,
) -> Result<Showcase> {
    let (newcomers, residents): (Vec<_>, Vec<_>) = specimen
        .photographs()
        .iter()
        .partition(|photograph| NEWCOMERS.contains(&photograph.file()));
    demand(
        newcomers.len() == NEWCOMERS.len(),
        "the specimen lacks its Persian or its Sphynx",
    )?;
    let _collection = testbed.create_private_dir(COLLECTION)?;
    for photograph in &residents {
        admit(testbed, photograph)?;
    }
    let app = testbed.launch(
        AppCommand::new(binary)
            .witness(WITNESS)
            .graphics(Graphics::Host)
            .network(Network::Deny)
            .runtime(Duration::from_secs(600))
            .arg(Testbed::guest_path(COLLECTION)),
    )?;
    let mut story: Duel<'_, '_> = Story::bind(
        testbed,
        &app,
        WindowQuery::title_exact(TITLE),
        ReactionBudget::functional(STARTUP),
    )?
    .with_observer(showcase)?
    .with_wait_timeout(STARTUP);
    let opened = story.ready(STARTUP)?;
    demand(
        opened.state.contract == UI_FINGERPRINT,
        format!("UI contract mismatch: {}", opened.state.contract),
    )?;
    let population = residents.len();
    let _residents = story.wait(Condition::new(
        "resident cats under comparison",
        move |state: &Observation| {
            !state.busy && state.pair_ready && state.visible_assets == population
        },
    ))?;

    story.chapter("Choose the stronger image")?;
    let mut duels = 0;
    for _resident_pair in 0..population.div_ceil(2) {
        duels += 1;
        judge(
            &mut story,
            if duels % 2 == 1 {
                Side::Left
            } else {
                Side::Right
            },
            duels,
        )?;
    }

    story.chapter("Rescan for newcomers")?;
    for photograph in &newcomers {
        admit(testbed, photograph)?;
    }
    let met = story
        .click(Target::Rescan)?
        .until(Condition::new(
            "the Persian meets the Sphynx",
            move |state: &Observation| {
                !state.busy
                    && state.visible_assets == population + NEWCOMERS.len()
                    && state.pair_files.as_ref().is_some_and(|files| {
                        files.iter().all(|file| NEWCOMERS.contains(&file.as_str()))
                    })
            },
        ))?
        .into_value();
    let persian = if met
        .state
        .pair_files
        .as_ref()
        .is_some_and(|[left, _]| left == PERSIAN)
    {
        Side::Left
    } else {
        Side::Right
    };
    freeze(&mut story, &ArtifactStem::FRONTISPIECE)?;

    story.chapter("Persian against Sphynx")?;
    let _favorite = story
        .click(Target::Favorite(persian))?
        .until(Condition::new("favorite Persian", |state: &Observation| {
            !state.busy && state.favorites == 1
        }))?;
    duels += 1;
    judge(&mut story, persian, duels)?;

    story.chapter("Browse the learned order")?;
    let browsing = story
        .click(Target::BrowseMode)?
        .until(Condition::new(
            "every visible tile developed at its resolution",
            |state: &Observation| {
                state.mode == "browse"
                    && !state.busy
                    && (state.thumbnails_faulted || state.tiles_wanting == Some(0))
            },
        ))?
        .into_value();
    demand(
        !browsing.state.thumbnails_faulted,
        format!("a specimen thumbnail faulted: {}", browsing.state.status),
    )?;
    freeze(&mut story, &ArtifactStem::new("learned-order")?)?;

    story.chapter("Inspect the leader")?;
    let _viewing = story
        .click(Target::BrowseTile)?
        .until(Condition::new("leader in full", |state: &Observation| {
            state.viewer_open && state.viewer_ready
        }))?;
    freeze(&mut story, &ArtifactStem::new("leader")?)?;

    let showcase = story.finish()?;
    app.terminate()?;
    Ok(showcase)
}

fn admit(testbed: &Testbed, photograph: &Photograph) -> Result<()> {
    let _admitted = testbed.write_private(
        Path::new(COLLECTION).join(photograph.file()),
        photograph.bytes(),
    )?;
    Ok(())
}

fn judge(story: &mut Duel<'_, '_>, side: Side, duels: u64) -> Result<()> {
    let _judged = story.click(Target::Choice(side))?.until(Condition::new(
        format!("duel {duels} judged"),
        move |state: &Observation| !state.busy && state.pair_ready && state.duels == duels,
    ))?;
    Ok(())
}

/// Develop a still once the pointer has left the window and the water has come
/// to rest, so that neither hover nor ripple marks it.
fn freeze(story: &mut Duel<'_, '_>, still: &ArtifactStem) -> Result<()> {
    let _calm = story
        .leave()?
        .until(Condition::new("water at rest", |state: &Observation| {
            state.water_calm
        }))?;
    story.still(still)
}
